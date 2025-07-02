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
    1) Диаризация (CPU)
    2) Whisper (попытка GPU FP16 → жёсткий fallback CPU float32)
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
    whisper_model = None

    # 2a) Попытка загрузить на GPU float16
    if settings.WHISPER_DEVICE.startswith("cuda"):
        try:
            whisper_model = WhisperModel(
                model_path,
                device=settings.WHISPER_DEVICE,    # e.g. "cuda"
                device_index=0,
                compute_type=settings.WHISPER_COMPUTE_TYPE  # "float16"
            )
            logger.info(f"✅ Loaded Whisper on GPU `{settings.WHISPER_DEVICE}` (compute_type={settings.WHISPER_COMPUTE_TYPE})")
        except Exception as gpu_err:
            logger.warning(
                f"⚠️ GPU load failed ({gpu_err}), falling back to CPU float32",
                exc_info=True
            )

    # 2b) Жёсткий fallback на CPU float32
    if whisper_model is None:
        try:
            whisper_model = WhisperModel(
                model_path,
                device="cpu",
                compute_type="default"  # float32
            )
            logger.info(f"✅ Loaded Whisper on CPU (compute_type=default)")
        except Exception as cpu_err:
            logger.error(f"❌ Failed to load Whisper model on CPU: {cpu_err}", exc_info=True)
            # Блокируем стартап, без модели бессмысленно продолжать
            raise

    _whisper = whisper_model

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
    Основная задача:
      1) Диаризация аудио
      2) Транскрипция сегментов Whisper
      3) Сохранение JSON и обновление статуса
    """
    session: AsyncSession = AsyncSessionLocal()
    try:
        # 0) Помечаем запись как "processing"
        await update_upload_status(session, upload_id, "processing")

        # 1) Диаризация
        diarization = _diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # 2) Транскрипция каждого сегмента
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

        # 3) Сохраняем результат в JSON-файл
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

        # 4) Обновляем статус на "completed" и удаляем временные файлы
        await update_upload_status(session, upload_id, "completed")
        cleanup_files(file_path, json_path)

    except Exception as e:
        logger.exception(f"Error in process_audio (upload_id={upload_id}): {e}")
        await update_upload_status(session, upload_id, "failed")
        raise
    finally:
        await session.close()

def cleanup_files(*paths: str):
    """
    Удаление временных файлов после обработки.
    """
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"Deleted file {p}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"Failed to delete {p}: {e}")