# tasks.py

import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from celery import Task
from celery_app import celery_app
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Lazy-loaded singletons to avoid re-loading large models per task invocation
# -----------------------------------------------------------------------------
_whisper_model: WhisperModel | None = None
_diarizer: Pipeline | None = None

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            model_size_or_path=settings.WHISPER_MODEL_NAME,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=settings.WHISPER_DEVICE_INDEX,
            inter_threads=settings.WHISPER_INTER_THREADS,
            intra_threads=settings.WHISPER_INTRA_THREADS,
        )
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
    """
    Perform speaker diarization on the entire file, chunked by DIARIZE_CHUNK_LENGTH seconds.
    Returns a list of segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    diarizer = get_diarizer()
    # Run diarization (the pipeline internally handles chunking if needed)
    timeline = diarizer(wav_path)
    segments = []
    for turn, _, speaker in timeline.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    # Optionally delete file after processing:
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
    # decode into successive segments
    segments, _info = model.transcribe(
        wav_path,
        beam_size=settings.WHISPER_BEAM_SIZE,
        best_of=settings.WHISPER_BEST_OF,
        return_timestamps=True,
        task=settings.WHISPER_TASK,  # e.g. "transcribe" or "translate"
        bahasa=None,
    )
    # segments is an iterable of (start, end, text)
    out = []
    for segment in segments:
        start, end, text = segment
        out.append({
            "start": float(start),
            "end": float(end),
            "text": text.strip(),
        })
    # Clean up if desired
    if settings.CLEAN_UP_UPLOADS:
        try:
            os.remove(wav_path)
        except Exception:
            pass
    return out