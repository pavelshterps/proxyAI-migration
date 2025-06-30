# tasks.py

import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import webrtcvad

from config.settings import settings

logger = logging.getLogger(__name__)

# Singletons so we only load each model once per worker process
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            f"Loading WhisperModel {{'path': '{settings.whisper_model_path}', "
            f"'device': '{settings.whisper_device}', 'compute_type': '{settings.whisper_compute_type}'}}"
        )
        _whisper_model = WhisperModel(
            str(settings.whisper_model_path),
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            device_index=settings.whisper_device_index,
        )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.makedirs(settings.diarizer_cache_dir, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache '{settings.diarizer_cache_dir}'")
        _diarizer = Pipeline.from_pretrained(
            settings.pyannote_protocol,
            cache_dir=str(settings.diarizer_cache_dir),
            use_auth_token=settings.huggingface_token
        )
        logger.info("Diarizer loaded")
    return _diarizer


def split_on_speech(audio_path: Path, aggressiveness: int = 3):
    """
    Use webrtcvad to split audio into contiguous speech segments.
    Returns a list of (start_sec, end_sec) tuples.
    """
    wav = AudioSegment.from_file(str(audio_path)).set_channels(1).set_frame_rate(16000)
    raw = wav.raw_data
    sample_rate = wav.frame_rate
    vad = webrtcvad.Vad(aggressiveness)

    frame_ms = 30
    frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit samples
    offsets = []
    is_speech = False
    segment_start = 0.0

    for i in range(0, len(raw), frame_bytes):
        chunk = raw[i : i + frame_bytes]
        timestamp = i / 2 / sample_rate  # seconds (bytes→samples)
        if len(chunk) < frame_bytes:
            break
        speech = vad.is_speech(chunk, sample_rate)
        if speech and not is_speech:
            is_speech = True
            segment_start = timestamp
        elif not speech and is_speech:
            is_speech = False
            offsets.append((segment_start, timestamp))
    # if file ends in speech
    if is_speech:
        offsets.append((segment_start, len(raw) / 2 / sample_rate))
    return offsets or [(0.0, len(raw) / 2 / sample_rate)]


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    """
    1) Splits audio on speech
    2) Transcribes each segment with Whisper
    3) Writes RESULTS_FOLDER/<upload_id>/transcript.json
    """
    whisper = get_whisper_model()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    dst = Path(settings.results_folder) / upload_id
    dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for '{src}'")
    segments = split_on_speech(src)
    logger.info(f"  → {len(segments)} speech segments")

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"  Segment {idx}: {start:.2f}s → {end:.2f}s")
        result = whisper.transcribe(
            str(src),
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language,
            vad_filter=True,
            word_timestamps=True,
            offset=start,
            duration=end - start,
        )
        text = result["segments"][0]["text"]
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text
        })

    out_path = dst / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Transcription complete, saved to '{out_path}'")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    """
    1) Runs speaker diarization on the whole file
    2) Writes RESULTS_FOLDER/<upload_id>/diarization.json
    """
    diarizer = get_diarizer()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    dst = Path(settings.results_folder) / upload_id
    dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for '{src}'")
    diarization = diarizer(str(src))

    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    out_path = dst / "diarization.json"
    out_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Diarization complete, saved to '{out_path}'")