# tasks.py
import os
from celery_app import celery_app
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# singletons
_diarizer = None
_whisper = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(settings.PYANNOTE_MODEL)
    return _diarizer

def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(
            settings.WHISPER_MODEL,
            device="cuda",
            compute_type="float16",
            inter_threads=1,
            intra_threads=1,
        )
    return _whisper

@celery_app.task(name="tasks.diarize_full")
def diarize_full(path: str):
    diarizer = get_diarizer()
    return diarizer({"audio": path})

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(segment_paths: list[str]):
    model = get_whisper()
    results = []
    for seg in segment_paths:
        segments, _ = model.transcribe(seg)
        results.append(segments)
    return results