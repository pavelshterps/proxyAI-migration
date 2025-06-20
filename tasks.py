import os
import time
import numpy as np
import whisperx
from celery import chord, Task
from celery.utils.log import get_task_logger
from .celery_app import celery_app
from .settings import (
    DEVICE, WHISPER_MODEL_NAME, WHISPER_COMPUTE_TYPE,
    ALIGN_MODEL_NAME, ALIGN_BEAM_SIZE, HUGGINGFACE_TOKEN
)
from .utils import load_audio, save_segments_to_db  # ваши утилиты

log = get_task_logger(__name__)

# кеш моделей выравнивания по языку
_align_model_cache: dict[str, tuple] = {}

def get_align_model(language: str):
    """Кэширование whisperx.align-модели по языку."""
    if language in _align_model_cache:
        return _align_model_cache[language]

    device = DEVICE if DEVICE == "cuda" else "cpu"
    try:
        model, metadata = whisperx.load_align_model(
            model_name_or_path=ALIGN_MODEL_NAME,
            device=device,
            language=language,
            beam_size=ALIGN_BEAM_SIZE,
            token=HUGGINGFACE_TOKEN or None
        )
    except Exception as e:
        log.warning(f"align-model load failed on {device}, fallback to cpu: {e}")
        model, metadata = whisperx.load_align_model(
            model_name_or_path=ALIGN_MODEL_NAME,
            device="cpu",
            language=language,
            beam_size=ALIGN_BEAM_SIZE,
        )
    _align_model_cache[language] = (model, metadata)
    return model, metadata

def estimate_processing_time(duration_sec: float, model_size: str) -> float:
    """Возвращает оценку времени в секундах (не мс)."""
    # грубая оценка: 1s аудио ~= 2s на модели large-v3 на GPU, 6s на CPU
    factor = 2 if DEVICE == "cuda" else 6
    return duration_sec * factor

@celery_app.task(bind=True, name="tasks.transcribe_chunk")
def transcribe_chunk(self: Task, audio_array: np.ndarray, start: float, end: float, language: str):
    model, metadata = get_align_model(language)
    try:
        aligned = whisperx.align(
            audio_array,        # сначала массив (np.ndarray)
            model,              # потом модель
            metadata,
            device=DEVICE,
        )
        return aligned
    except Exception as exc:
        log.error(f"align failed: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=30, max_retries=3)

@celery_app.task(name="tasks.transcribe_task")
def transcribe_task(filepath: str):
    # 1. загрузить и разбить на отрезки
    audio_array, sr = load_audio(filepath)
    duration = audio_array.shape[0] / sr
    estimate = estimate_processing_time(duration, WHISPER_MODEL_NAME)

    segments = whisperx.transcribe(
        model=WHISPER_MODEL_NAME,
        compute_type=WHISPER_COMPUTE_TYPE,
        audio_filepath=filepath,
        device=DEVICE,
    )["segments"]

    # 2. Запустить выравнивание по частям
    subtasks = [
        transcribe_chunk.s(audio_array[slice_.start:slice_.end], slice_.start, slice_.end, seg.language)
        for seg in segments
        for slice_ in [seg.time_frame]  # время сегмента
    ]
    callback = merge_chunks.s(filepath)
    chord_id = chord(subtasks)(callback).id

    return {
        "task_id": transcribe_task.request.id,
        "estimate": estimate,
        "merge_task_id": chord_id
    }

@celery_app.task(name="tasks.merge_chunks")
def merge_chunks(aligned_results: list, filepath: str):
    # объединить результаты, сохранить в БД и файл
    final_segments = save_segments_to_db(aligned_results, filepath)
    audio_url = os.path.basename(filepath)
    return {
        "status": "SUCCESS",
        "segments": final_segments,
        "audio_filepath": f"/files/{audio_url}"
    }