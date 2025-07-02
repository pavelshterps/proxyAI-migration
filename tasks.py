# tasks.py

import os
import json
import logging

from celery import Celery, signals
from celery.utils.log import get_task_logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline

from config.settings import settings
from crud import update_upload_status

logger = get_task_logger(__name__)

# Инициализация Celery через внешний файл config/celery.py
app = Celery("proxyai")
app.config_from_object("config.celery")

# Асинхронный движок и фабрика сессий SQLAlchemy
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

# Глобальные объекты моделей
_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Предзагрузка моделей при старте каждого воркера:
      1) Диаризация (always CPU)
      2) Whisper:
           a) GPU FP16 (если настроено)
           b) иначе CPU default (int8/float32)
    """
    global _whisper, _diarizer

    # 1) Диаризация на CPU
    try:
        _diarizer = DiarizationPipeline.from_pretrained(settings.PYANNOTE_PIPELINE)
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline: {e}", exc_info=True)
        raise

    model_path = settings.WHISPER_MODEL_PATH

    # Попытка №1: загрузить на GPU FP16
    if settings.WHISPER_DEVICE.startswith("cuda"):
        try:
            _whisper = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16",
                device_index=0,
            )
            logger.info(f"✅ Loaded Whisper FP16 on GPU from `{model_path}`")
            return
        except Exception as gpu_exc:
            logger.warning(
                "⚠️ GPU Whisper load failed, falling back to CPU: "
                f"{type(gpu_exc).__name__}: {gpu_exc}"
            )

    # Попытка №2: загрузить на CPU с дефолтным форматом (инт8/float32)
    try:
        _whisper = WhisperModel(
            model_path,
            device="cpu",
            compute_type="default",
        )
        logger.info(f"✅ Loaded Whisper on CPU (`compute_type=default`) from `{model_path}`")
    except Exception as cpu_exc:
        logger.error(
            "❌ Both GPU and CPU Whisper load failed:\n"
            f"  GPU error: {gpu_exc}\n"
            f"  CPU error: {cpu_exc}",
            exc_info=True
        )
        raise

@app.task(
    bind=True,
    name="process_audio",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
async def process_audio(self, upload_id: int, file_path: str):
    """
    1) Диаризация
    2) Транскрипция сегментов Whisper
    3) Сохранение JSON
    4) Обновление статуса и очистка
    """
    session: AsyncSession = AsyncSessionLocal()
    try:
        await update_upload_status(session, upload_id, "processing")

        # Диаризация
        diarization = _diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # Транскрипция
        transcriptions = []
        for seg in segments:
            result = _whisper.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=False,
                segment=seg
            )
            text = " ".join([chunk.text for chunk in result])
            transcriptions.append({**seg, "text": text})

        # Сохраняем JSON
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

        await update_upload_status(session, upload_id, "completed")
        cleanup_files(file_path, json_path)

    except Exception:
        logger.exception(f"Error in process_audio (upload_id={upload_id})")
        await update_upload_status(session, upload_id, "failed")
        raise
    finally:
        await session.close()

def cleanup_files(*paths: str):
    """Удаляем временные файлы после обработки."""
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"Deleted file {p}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"Failed to delete {p}: {e}")