# tasks.py

import os
import logging
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from celery import Task
from celery_app import celery_app
from config.settings import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Lazy-loaded singletons to avoid re-loading large models per task invocation
# -----------------------------------------------------------------------------
_whisper_model: WhisperModel | None = None
_diarizer: Pipeline | None = None

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        params = {
            "model_size_or_path": settings.WHISPER_MODEL_NAME,
            "device": settings.WHISPER_DEVICE,
            "compute_type": settings.WHISPER_COMPUTE_TYPE,
            "device_index": settings.WHISPER_DEVICE_INDEX,
        }
        logger.info(f"Loading WhisperModel with params: {params}")
        _whisper_model = WhisperModel(**params)
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_MODEL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.HF_CACHE_DIR,
        )
    return _diarizer

# -----------------------------------------------------------------------------
# Celery tasks
# -----------------------------------------------------------------------------

@celery_app.task(
    name="tasks.diarize_full",
    bind=True,
    acks_late=True,
    ignore_result=False,
)
def diarize_full(self: Task, wav_path: str) -> list[dict]:
    logger.info(f"Starting diarize_full on {wav_path}")
    diarizer = get_diarizer()
    timeline = diarizer(wav_path)
    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in timeline.itertracks(yield_label=True)
    ]

    logger.info(f"Scheduling transcribe_segments for {wav_path}")
    transcribe_segments.apply_async(
        args=(wav_path,),
        queue="preprocess_gpu"
    )

    if settings.CLEAN_UP_UPLOADS:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    return segments

@celery_app.task(
    name="tasks.transcribe_segments",
    bind=True,
    acks_late=True,
    ignore_result=False,
)
def transcribe_segments(self: Task, wav_path: str) -> list[dict]:
    logger.info(f"Starting transcribe_segments on {wav_path}")
    """
    Transcribe the given audio file with Whisper into segments.
    Returns a list of {"start": float, "end": float, "text": str}.
    """
    model = get_whisper_model()
    segments, _info = model.transcribe(
        wav_path,
        beam_size=settings.WHISPER_BEAM_SIZE,
        best_of=settings.WHISPER_BEST_OF,
        task=settings.WHISPER_TASK,
    )
    out = []
    for start, end, text in segments:
        out.append({
            "start": float(start),
            "end": float(end),
            "text": text.strip(),
        })

    if settings.CLEAN_UP_UPLOADS:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    return out