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

# Настройка Celery
app = Celery("proxyai")
app.config_from_object("config.celery")

# Логгер задач
logger = get_task_logger(__name__)

# Асинхронный движок и сессии SQLAlchemy
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

# Глобальные ссылки на модели
_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None


def get_whisper() -> WhisperModel:
    """Вернуть предварительно загруженный Whisper; бросить, если не загружен."""
    if _whisper is None:
        raise RuntimeError("Whisper model is not loaded")
    return _whisper


def get_diarizer() -> DiarizationPipeline:
    """Вернуть предварительно загруженный DiarizationPipeline; бросить, если не загружен."""
    if _diarizer is None:
        raise RuntimeError("Diarizer pipeline is not loaded")
    return _diarizer


@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """Сигнал Celery: при старте воркера загружаем модели."""
    global _whisper, _diarizer

    # 1) Диаризация на CPU
    try:
        _diarizer = DiarizationPipeline.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline `{settings.PYANNOTE_PIPELINE}`: {e}")
        raise

    # 2) Whisper: сначала GPU+float16, иначе CPU+int8
    model_path = settings.WHISPER_MODEL_PATH  # локальный путь к quantised-модели
    whisper_kwargs = {
        "device": settings.WHISPER_DEVICE,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
        "batch_size": settings.WHISPER_BATCH_SIZE,
    }
    if settings.HUGGINGFACE_CACHE_DIR:
        whisper_kwargs["cache_dir"] = settings.HUGGINGFACE_CACHE_DIR

    try:
        _whisper = WhisperModel(model_path, **whisper_kwargs)
        logger.info(f"✅ Loaded Whisper model on {settings.WHISPER_DEVICE} ({settings.WHISPER_COMPUTE_TYPE})")
    except Exception as e_gpu:
        logger.warning(f"❌ Failed to load Whisper on {settings.WHISPER_DEVICE}: {e_gpu}")
        # Падение на CPU + int8
        try:
            cpu_kwargs = whisper_kwargs.copy()
            cpu_kwargs.update({"device": "cpu", "compute_type": "int8"})
            _whisper = WhisperModel(model_path, **cpu_kwargs)
            logger.info("✅ Loaded Whisper model on CPU (int8 fallback)")
        except Exception as e_cpu:
            logger.error(f"❌ Failed to load Whisper on CPU fallback: {e_cpu}")
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
     1) обновить статус → processing
     2) диаризовать
     3) транскрибировать каждый сегмент
     4) сохранить JSON
     5) обновить статус → completed / failed
     6) удалить временные файлы
    """
    session = AsyncSessionLocal()
    try:
        await update_upload_status(session, upload_id, "processing")

        # 1) Диаризация
        diarization = get_diarizer()({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # 2) Транскрипция
        whisper = get_whisper()
        transcriptions = []
        for seg in segments:
            result = whisper.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=False,
                segment=seg
            )
            text = " ".join([s.text for s in result])
            transcriptions.append({**seg, "text": text})

        # 3) Запись в JSON
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

        # 4) Завершение
        await update_upload_status(session, upload_id, "completed")
        cleanup_files(file_path, json_path)

    except Exception as e:
        logger.exception(f"Error in process_audio (upload_id={upload_id}): {e}")
        await update_upload_status(session, upload_id, "failed")
        raise
    finally:
        await session.close()


def cleanup_files(*paths: str):
    """Пытаемся удалить указанные файлы, логируем результат."""
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"Deleted file {p}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"Failed to delete {p}: {e}")