# tasks.py

import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

from config.settings import settings

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            f"Loading WhisperModel "
            f"{{'path': '{settings.whisper_model_path}', 'device': '{settings.whisper_device}', 'compute': '{settings.whisper_compute_type}'}}"
        )
        _whisper_model = WhisperModel(
            str(settings.whisper_model_path),
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            device_index=settings.whisper_device_index
        )
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.diarizer_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache '{cache_dir}'")
        _diarizer = Pipeline.from_pretrained(
            settings.pyannote_protocol,
            cache_dir=str(cache_dir)
        )
        logger.info("Diarizer loaded")
    return _diarizer

def split_audio_fixed_windows(audio_path: Path):
    window_s = settings.segment_length_s
    audio = AudioSegment.from_file(str(audio_path))
    total_ms = len(audio)
    win_ms = window_s * 1000
    segments = []
    for start_ms in range(0, total_ms, win_ms):
        end_ms = min(start_ms + win_ms, total_ms)
        segments.append((start_ms/1000.0, end_ms/1000.0))
    return segments

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    dst_dir = Path(settings.results_folder) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for '{src}'")
    segments = split_audio_fixed_windows(src)
    logger.info(f"  -> {len(segments)} segments of up to {settings.segment_length_s}s")

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"  Segment {idx}: {start:.1f}s → {end:.1f}s")
        res = whisper.transcribe(
            str(src),
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language,
            vad_filter=True,
            word_timestamps=True,
            offset=start,
            duration=end - start,
        )
        text = res["segments"][0]["text"]
        transcript.append({"segment": idx, "start": start, "end": end, "text": text})

    out_path = dst_dir / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Transcription complete: saved to '{out_path}'")

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    dst_dir = Path(settings.results_folder) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for '{src}'")
    diarization = diarizer(str(src))

    tracks = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        tracks.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(tracks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Diarization complete: saved to '{out_path}'")