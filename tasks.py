# tasks.py
import os
from celery_app import celery_app
from config.settings import settings
from pathlib import Path

# pyannote diarizer
from pyannote.audio import Pipeline
# faster-whisper
from faster_whisper import WhisperModel

CACHE_DIR = os.getenv("HF_CACHE", "/hf_cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

_diarizer = None
def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=CACHE_DIR,
        )
    return _diarizer

_whisper = None
def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(
            model=settings.WHISPER_MODEL,
            device=settings.DEVICE,
            device_index=0,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            cache_dir=CACHE_DIR,
            inter_threads=1,
            intra_threads=1,
        )
    return _whisper

@celery_app.task(name="tasks.diarize_full")
def diarize_full(filepath: str):
    diarizer = get_diarizer()
    diarization = diarizer(filepath)
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": spk}
        for turn, _, spk in diarization.itertracks(yield_label=True)
    ]
    # fan-out each segment for transcription
    for seg in segments:
        celery_app.send_task(
            "tasks.transcribe_segments",
            args=(filepath, seg["start"], seg["end"]),
        )
    return segments

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(filepath: str, start: float, end: float):
    model = get_whisper()
    segments, _ = model.transcribe(
        filepath,
        batch_size=1,
        beam_size=settings.ALIGN_BEAM_SIZE,
        chunk_length_s=30,
        chunk_overlap_s=5,
        condition_on_previous_text=True,
        word_timestamps=True,
    )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]