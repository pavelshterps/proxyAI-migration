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
    1) Диаризация
    2) Whisper (FP16 на GPU или CPU, с учётом локального кеша)
    """
    global _whisper, _diarizer

    # 1) Диаризация на CPU
    try:
        _diarizer = DiarizationPipeline.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.DIARIZER_CACHE_DIR or "/data/diarizer_cache"
        )
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline: {e}", exc_info=True)
        raise

    # 2) Whisper (FP16 на GPU или CPU)
    model_path = settings.WHISPER_MODEL_PATH
    whisper_kwargs = {
        "device": settings.WHISPER_DEVICE,
        "device_index": 0,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,  # float16
        "inter_threads": (
            settings.GPU_CONCURRENCY
            if settings.WHISPER_DEVICE.lower().startswith("cuda")
            else settings.CPU_CONCURRENCY
        ),
        "intra_threads": (
            settings.CPU_CONCURRENCY
            if settings.WHISPER_DEVICE.lower() == "cpu"
            else 0
        ),
        # указываем, где хранится локальный кеш huggingface
        "cache_dir": settings.HUGGINGFACE_CACHE_DIR or "/hf_cache",
    }

    try:
        _whisper = WhisperModel(model_path, **whisper_kwargs)
        logger.info(f"✅ Loaded Whisper model from `{model_path}` on `{settings.WHISPER_DEVICE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}", exc_info=True)
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
            # faster-whisper возвращает список маленьких сегментов
            result = _whisper.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=False,
                segment=seg
            )
            # Собираем текст из кусочков
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