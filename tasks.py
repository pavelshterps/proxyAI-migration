# tasks.py
import os
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from celery_app import celery_app
from config.settings import settings

# Синглтоны для моделей
_diarizer = None
_whisper = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.environ["HF_HOME"] = settings.HF_CACHE
        _diarizer = Pipeline.from_pretrained(settings.PYANNOTE_MODEL)
    return _diarizer

def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(
            settings.WHISPER_MODEL,
            device="cuda",
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=0,
            inter_threads=1,
            intra_threads=1,
        )
    return _whisper

@celery_app.task(name="tasks.diarize_full")
def diarize_full(path: str):
    diarizer = get_diarizer()
    # Возвращаем сегменты speakers
    return [
        {"start": turn.start, "end": turn.end, "speaker": turn.label}
        for turn in diarizer(path).get_timeline()
    ]

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(path: str):
    whisper = get_whisper()
    # Разбиение на чанки
    segments, _ = whisper.transcribe(
        path,
        beam_size=5,
        word_timestamps=False,
        language=None,
        vad_filter=False,
        max_initial_timestamp=0.0,
        chunk_length_s=settings.CHUNK_LENGTH_S,
    )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]