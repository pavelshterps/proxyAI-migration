# tasks.py
import os
import json
import logging
from pathlib import Path

import structlog
from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
from vad import VoiceActivityDetector  # assume we added a VAD helper

from config.settings import settings

logger = structlog.get_logger()

_whisper = None
_diarizer = None

def get_whisper_model():
    global _whisper
    if _whisper is None:
        logger.info("Loading WhisperModel", path=str(settings.whisper_model_path))
        _whisper = WhisperModel(
            str(settings.whisper_model_path),
            device=settings.whisper_device,
            device_index=settings.whisper_device_index,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("WhisperModel ready")
    return _whisper

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        Path(settings.diarizer_cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Loading Diarizer", protocol=settings.pyannote_protocol)
        _diarizer = Pipeline.from_pretrained(
            settings.pyannote_protocol,
            cache_dir=str(settings.diarizer_cache_dir),
            use_auth_token=settings.huggingface_token,
        )
        logger.info("Diarizer ready")
    return _diarizer

def split_audio_segments(audio_path: Path):
    # VAD‐based chunking
    detector = VoiceActivityDetector(str(audio_path))
    speech_regions = detector.detect_speech()
    segments = []
    for (start_ms, end_ms) in speech_regions:
        segments.append((start_ms/1000, end_ms/1000))
    # fallback to fixed windows if no speech
    if not segments:
        total = AudioSegment.from_file(str(audio_path)).duration_seconds
        window = settings.segment_length_s
        segments = [(i, min(i+window, total)) for i in range(0, int(total), window)]
    return segments

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    out_dir = Path(settings.results_folder) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Transcription started", upload=upload_id)
    segments = split_audio_segments(src)
    transcript = []

    for idx, (start, end) in enumerate(segments):
        logger.debug("Transcribing", segment=idx, start=start, end=end)
        res = whisper.transcribe(
            str(src),
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language,
            vad_filter=False,  # we pre‐segmented
            offset=start,
            duration=end - start,
            word_timestamps=True,
        )
        text = "".join(seg["text"] for seg in res["segments"])
        transcript.append(dict(segment=idx, start=start, end=end, text=text))

    (out_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    logger.info("Transcription saved", path=str(out_dir / "transcript.json"))

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    out_dir = Path(settings.results_folder) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Diarization started", upload=upload_id)
    result = diarizer(str(src))
    speakers = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in result.itertracks(yield_label=True)
    ]
    (out_dir / "diarization.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=2))
    logger.info("Diarization saved", path=str(out_dir / "diarization.json"))