import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from celery_app import celery_app
from config.settings import settings

# ленивое создание моделей
_diarizer: Pipeline | None = None
_whisper: WhisperModel | None = None

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        os.environ["HF_HOME"] = "/hf_cache"
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
    return _diarizer

def get_model() -> WhisperModel:
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            cache_dir="/hf_cache",
            device_index=0,
            inter_threads=1,
            intra_threads=1,
        )
    return _whisper

@celery_app.task(name="tasks.diarize_full")
def diarize_full(filepath: str) -> list[dict]:
    diarizer = get_diarizer()
    # получаем сегменты с метками спикеров
    return [{"start": turn.start, "end": turn.end, "speaker": turn.label}
            for turn in diarizer(filepath)]

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(segment: dict) -> dict:
    model = get_model()
    result = model.transcribe(
        segment["file"],
        beam_size=settings.ALIGN_BEAM_SIZE,
        language="en",
        without_timestamps=False,
    )
    return {"text": result[0].text, **segment}