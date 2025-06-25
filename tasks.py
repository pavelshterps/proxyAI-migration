# tasks.py
from celery_app import celery_app
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from config.settings import settings

# lazy‐load both models once, on first use:
diarizer = None
def get_diarizer():
    global diarizer
    if diarizer is None:
        diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_MODEL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.HF_HOME,
        )
    return diarizer

whisper_model = None
def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device="cuda",
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=0,
            inter_threads=1,
            intra_threads=1,
            cache_dir=settings.HF_HOME,
        )
    return whisper_model

@celery_app.task(name="tasks.diarize_full")
def diarize_full(filepath: str):
    model = get_diarizer()
    return model(filepath)

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(filepath: str):
    model = get_whisper()
    return model.transcribe(
        filepath,
        beam_size=settings.ALIGN_BEAM_SIZE,
    )