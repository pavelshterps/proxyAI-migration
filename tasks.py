import os
import numpy as np
import whisperx
from celery import chord, Task
from celery.utils.log import get_task_logger
from celery_app import celery_app
from settings import DEVICE, WHISPER_MODEL_NAME, WHISPER_COMPUTE_TYPE, ALIGN_MODEL_NAME, ALIGN_BEAM_SIZE, HUGGINGFACE_TOKEN
from utils import load_audio, save_segments_to_db

log = get_task_logger(__name__)

# Кэш моделей по языку
_align_cache: dict[str, tuple] = {}

def get_align_model(language: str):
    if language in _align_cache:
        return _align_cache[language]

    dev = DEVICE if DEVICE == "cuda" else "cpu"
    try:
        model, metadata = whisperx.load_align_model(
            ALIGN_MODEL_NAME,
            dev,
            language,
            ALIGN_BEAM_SIZE,
            token=HUGGINGFACE_TOKEN or None
        )
    except Exception as e:
        log.warning(f"Fallback to CPU for align-model ({e})")
        model, metadata = whisperx.load_align_model(
            ALIGN_MODEL_NAME,
            "cpu",
            language,
            ALIGN_BEAM_SIZE,
        )
    _align_cache[language] = (model, metadata)
    return model, metadata

def estimate_processing_time(duration_sec: float) -> float:
    # секунда аудио → 2s на GPU или 6s на CPU
    factor = 2 if DEVICE == "cuda" else 6
    return duration_sec * factor

@celery_app.task(bind=True, name="tasks.transcribe_chunk")
def transcribe_chunk(self: Task, audio_array: np.ndarray, start: float, end: float, language: str):
    model, metadata = get_align_model(language)
    try:
        aligned = whisperx.align(
            audio_array,
            model,
            metadata,
            device=DEVICE,
        )
        return aligned
    except Exception as exc:
        log.error(f"align failed: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=30, max_retries=3)

@celery_app.task(name="tasks.transcribe_task")
def transcribe_task(filepath: str):
    audio_array, sr = load_audio(filepath)
    duration = audio_array.shape[0] / sr
    estimate = estimate_processing_time(duration)

    segments = whisperx.transcribe(
        model=WHISPER_MODEL_NAME,
        compute_type=WHISPER_COMPUTE_TYPE,
        audio_filepath=filepath,
        device=DEVICE,
    )["segments"]

    subtasks = [
        transcribe_chunk.s(
            audio_array[int(seg.start * sr):int(seg.end * sr)],
            seg.start,
            seg.end,
            seg.language
        )
        for seg in segments
    ]
    merge_id = chord(subtasks)(merge_chunks.s(filepath)).id

    return {
        "task_id": transcribe_task.request.id,
        "estimate": estimate,
        "merge_task_id": merge_id
    }

@celery_app.task(name="tasks.merge_chunks")
def merge_chunks(aligned_results: list, filepath: str):
    final_segments = save_segments_to_db(aligned_results, filepath)
    filename = os.path.basename(filepath)
    return {
        "status": "SUCCESS",
        "segments": final_segments,
        "audio_filepath": f"/files/{filename}"
    }