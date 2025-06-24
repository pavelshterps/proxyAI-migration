import os
from celery_app import celery_app
from config.settings import (
    DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_MODEL,
    ALIGN_BEAM_SIZE,
    PYANNOTE_PROTOCOL,
)
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# однократные загрузки моделей
_diarizer = None
_whisper_model = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        token = os.getenv("HUGGINGFACE_TOKEN")
        _diarizer = Pipeline.from_pretrained(PYANNOTE_PROTOCOL, use_auth_token=token)
    return _diarizer

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        cache_dir = os.getenv("HF_HOME", "/tmp/hf_cache")
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cache_dir=cache_dir
        )
    return _whisper_model

@celery_app.task
def diarize_full(filepath):
    pipeline = get_diarizer()
    output = pipeline(filepath)
    segments = []
    for turn, _, speaker in output.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments

@celery_app.task
def transcribe_segments(filepath):
    model = get_whisper_model()
    text_segments, _ = model.transcribe(filepath, beam_size=ALIGN_BEAM_SIZE)
    return {"text": " ".join([seg.text for seg in text_segments])}

@celery_app.task
def transcribe_full(filepath):
    # Сначала диаризация
    segments = diarize_full(filepath)
    model = get_whisper_model()
    full_text = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        text_segments, _ = model.transcribe(
            filepath,
            beam_size=ALIGN_BEAM_SIZE,
            segment_timestamps=[(start, end)]
        )
        # добавляем первый (и единственный) фрагмент
        full_text.append(text_segments[0].text)
    return {"text": " ".join(full_text), "segments": segments}