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

# инициализация Celery-приложения
app = Celery("proxyai")
app.config_from_object("config.celery")

# Асинхронный движок SQLAlchemy и фабрика сессий
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Загружаем пайплайн Pyannote на CPU и quantized Whisper на GPU
    из локального кэша без повторной загрузки из HuggingFace.
    """
    global _whisper, _diarizer

    # 1) Диаризация на CPU
    try:
        _diarizer = DiarizationPipeline.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline: {e}")
        raise

    # 2) Whisper на GPU (quantized модель уже в settings.WHISPER_MODEL_PATH)
    model_path = settings.WHISPER_MODEL_PATH
    whisper_init_kwargs = {
        "device": settings.WHISPER_DEVICE,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
        # ctranslate2.models.Whisper __init__ не принимает batch_size и cache_dir
    }
    try:
        _whisper = WhisperModel(model_path, **whisper_init_kwargs)
        logger.info(f"✅ Loaded Whisper model from `{model_path}`")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
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
    1) обновить статус на processing,
    2) прогнать диаризацию,
    3) прогнать транскрипцию каждого сегмента с batch_size,
    4) сохранить результаты в JSON,
    5) обновить статус на completed/failed,
    6) удалить файлы.
    """
    session = AsyncSessionLocal()
    json_path = f"{file_path}.json"
    try:
        # статус → processing
        await update_upload_status(session, upload_id, "processing")

        # 1) Диаризация
        diarization = _diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # 2) Транскрипция с batch_size
        transcriptions = []
        for seg in segments:
            result = _whisper.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=False,
                segment=seg,
                batch_size=settings.WHISPER_BATCH_SIZE,
            )
            text = " ".join([s.text for s in result])
            transcriptions.append({**seg, "text": text})

        # 3) Сохранение JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)
        logger.info(f"📄 Saved transcription JSON to {json_path}")

        # 4) Успешное завершение
        await update_upload_status(session, upload_id, "completed")
        logger.info(f"✅ Upload {upload_id} completed")

    except Exception as e:
        logger.exception(f"🔥 Error in process_audio (upload_id={upload_id}): {e}")
        # статус → failed
        try:
            await update_upload_status(session, upload_id, "failed")
        except Exception as ee:
            logger.error(f"❌ Failed to mark upload {upload_id} as failed: {ee}")
        raise

    finally:
        # Всегда удаляем исходный файл и JSON
        cleanup_files(file_path, json_path)
        await session.close()

def cleanup_files(*paths: str):
    """Удалить файлы по списку путей, залогировать результат."""
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"🗑️ Deleted file {p}")
        except FileNotFoundError:
            logger.warning(f"⚠️ File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete {p}: {e}")