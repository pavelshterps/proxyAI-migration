# tasks.py

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import structlog
from celery import Celery
from celery.utils.log import get_task_logger
from fastapi import HTTPException
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline
from pydantic.error_wrappers import ValidationError

from config.settings import settings

# Инициализация Celery
app = Celery("proxyai")
app.conf.broker_url = settings.CELERY_BROKER_URL
app.conf.result_backend = settings.CELERY_RESULT_BACKEND
app.conf.task_queues = {
    "preprocess_cpu": {"exchange": "preprocess_cpu", "binding_key": "preprocess_cpu"},
}

logger = structlog.get_logger()
task_logger = get_task_logger(__name__)

# Singleton-модели
_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None


def get_whisper() -> WhisperModel:
    """
    Возвращает единственный экземпляр WhisperModel.
    Загружает модель по её HF-идентификатору, используя указанный cache_dir
    и compute_type.
    """
    global _whisper
    if _whisper is None:
        model_id = settings.WHISPER_MODEL_PATH  # например "guillaumekln/faster-whisper-medium"
        init_kwargs: dict[str, str] = {
            "device": settings.WHISPER_DEVICE,
        }
        # если указан compute_type (int8, fp16 и т.п.)
        if settings.WHISPER_COMPUTE_TYPE:
            init_kwargs["compute_type"] = settings.WHISPER_COMPUTE_TYPE
        # если задан каталог кеша HF, передаём его
        if settings.HUGGINGFACE_CACHE_DIR:
            init_kwargs["cache_dir"] = settings.HUGGINGFACE_CACHE_DIR

        logger.info("loading whisper model", model=model_id, **init_kwargs)
        _whisper = WhisperModel(model_id, **init_kwargs)
        logger.info("whisper model loaded", model=model_id)

    return _whisper


def get_diarizer() -> DiarizationPipeline:
    """
    Возвращает единственный экземпляр пайплайна диаризации pyannote.
    """
    global _diarizer
    if _diarizer is None:
        try:
            _diarizer = DiarizationPipeline.from_pretrained(
                settings.PYANNO_PIPELINE,
                use_auth_token=settings.HUGGINGFACE_TOKEN,
            )
        except Exception as e:
            task_logger.error("Failed to load diarization pipeline", error=str(e))
            raise
    return _diarizer


@app.task(name="tasks.cleanup_old_files")
def cleanup_old_files(path: str):
    """
    Удаляет временные файлы по указанному пути, если они старше retention.
    """
    try:
        shutil.rmtree(path)
        logger.info("cleaned up files", path=path)
    except Exception as e:
        logger.warning("failed to cleanup files", path=path, error=str(e))


@app.task(name="tasks.transcribe_segments", bind=True)
def transcribe_segments(self, audio_path: str, segments: list[tuple[float, float]]) -> list[dict]:
    """
    Транскрибирует список сегментов аудио с помощью WhisperModel.
    """
    whisper = get_whisper()
    results: list[dict] = []
    try:
        # faster-whisper возвращает (segment, text, logprob, tokens) для каждого сегмента
        for start, end in segments:
            segment = whisper.transcribe(
                audio_path,
                beam_size=settings.WHISPER_BEAM_SIZE,
                initial_prompt=None,
                language=None,
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=True,
                prompt=None,
                segment_offsets=[(start, end)],
            )
            results.append({
                "start": start,
                "end": end,
                "text": segment[0].text,
                "tokens": segment[0].tokens,
                "logprob": segment[0].avg_logprob,
            })
    except Exception as e:
        task_logger.error("Whisper transcription failed", error=str(e))
        raise HTTPException(status_code=500, detail="Transcription error")
    return results


@app.task(name="tasks.diarize_full")
def diarize_full(audio_path: str) -> list[dict]:
    """
    Диаризует всё аудио, возвращая список сегментов с канальным указанием спикера.
    """
    pipeline = get_diarizer()
    try:
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        return segments
    except ValidationError as e:
        task_logger.error("Diarization validation failed", error=str(e))
        raise HTTPException(status_code=400, detail="Diarization pipeline error")
    except Exception as e:
        task_logger.error("Diarization failed", error=str(e))
        raise HTTPException(status_code=500, detail="Diarization error")


# Планировщик (beat) может посылать этот таск по расписанию
@app.task(name="tasks.cleanup_all_old_files")
def cleanup_all_old_files():
    """
    Тут можно обойти все рабочие директории и удалить всё старое.
    """
    base = Path(settings.UPLOAD_FOLDER)
    for sub in base.iterdir():
        if sub.is_dir():
            cleanup_old_files.delay(str(sub))