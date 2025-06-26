# tasks.py

from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from celery_app import celery_app
from config.settings import settings
import torch

_diarizer_pipeline = None

def get_diarizer_pipeline():
    global _diarizer_pipeline
    if _diarizer_pipeline is None:
        _diarizer_pipeline = Pipeline.from_pretrained(settings.PYANNOTE_MODEL)
    # Only move to CUDA if available; otherwise stay on CPU
    desired_device = settings.WHISPER_DEVICE  # e.g., "cuda" or "cpu"
    device = desired_device if desired_device == "cuda" and torch.cuda.is_available() else "cpu"
    _diarizer_pipeline.to(device)
    return _diarizer_pipeline

@celery_app.task(name="tasks.diarize_full")
def diarize_full(wav_path: str):
    pipeline = get_diarizer_pipeline()
    # ... rest of your logic, splitting into 30-second chunks, etc.