# tasks.py
import os
from celery_app import celery_app
from config.settings import settings
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

@celery_app.task(name="tasks.diarize_full")
def diarize_full(filepath: str):
    # full-file diarization
    diarizer = Pipeline.from_pretrained(
        settings.PYANNOTE_PROTOCOL,
        use_auth_token=settings.HUGGINGFACE_TOKEN,
    )
    diarization = diarizer(filepath)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })
    return segments

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(filepath: str):
    # chunked GPU Whisper
    model = WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        device_index=0,
        inter_threads=1,
        intra_threads=1,
        max_cached_batches=1
    )
    result = []
    for seg in model.transcribe(
        filepath,
        beam_size=settings.ALIGN_BEAM_SIZE,
        chunk_length_s=30
    ):
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })
    return result