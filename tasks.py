# tasks.py

import os
import json
import logging
from celery import signals
from celery.utils.log import get_task_logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline

from config.settings import settings
from crud import update_upload_status

logger = get_task_logger(__name__)

# ---------- SQLAlchemy setup ----------
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

# ---------- Model handles ----------
_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Предзагрузка моделей при старте Celery-воркера.
    Диаризация идет на CPU, транскрипция на GPU через очередь.
    """
    global _whisper, _diarizer

    # --- 1) Диаризация на CPU ---
    try:
        _diarizer = DiarizationPipeline.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"Diarizer loaded: {settings.PYANNOTE_PIPELINE}")
    except Exception as e:
        logger.error(f"Failed to load diarization pipeline `{settings.PYANNOTE_PIPELINE}`: {e}")
        raise

    # --- 2) Whisper на GPU, локальный quantized model path ---
    model_path = settings.WHISPER_MODEL_PATH  # ожидаем здесь уже готовый квантизированный каталог
    whisper_kwargs = {
        "device": settings.WHISPER_DEVICE,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
        # ctranslate2 Whiper не поддерживает batch_size и cache_dir в конструкторе
    }

    try:
        _whisper = WhisperModel(model_path, **whisper_kwargs)
        logger.info(f"Whisper model loaded from `{model_path}`")
    except Exception as e:
        logger.error(f"Cannot load Whisper model from `{model_path}`: {e}")
        raise

def get_whisper() -> WhisperModel:
    """Возвращает предзагруженную WhisperModel."""
    if _whisper is None:
        preload_and_warmup()
    return _whisper

def get_diarizer() -> DiarizationPipeline:
    """Возвращает предзагруженный DiarizationPipeline."""
    if _diarizer is None:
        preload_and_warmup()
    return _diarizer

# ---------- Основная цепочка задач ----------
@app.task(
    bind=True,
    name="diarize_full",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
async def diarize_full(self, upload_id: int, file_path: str):
    """
    1) Диаризация CAП файла → выдаёт список сегментов с {start, end, speaker}.
    2) Сразу же пушит transcribe_segments в GPU-очередь.
    """
    session = AsyncSessionLocal()
    try:
        await update_upload_status(session, upload_id, "processing")
        diarizer = get_diarizer()
        diarization = diarizer({"uri": file_path, "audio": file_path})

        segments: list[dict] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker  # e.g. "SPEAKER_00"
            })

        # передаём на транскрипцию
        from celery_app import celery_app
        celery_app.send_task(
            "transcribe_segments",
            args=(upload_id, file_path, segments),
            queue="preprocess_gpu"
        )

    except Exception as e:
        logger.exception(f"Error in diarize_full (upload_id={upload_id}): {e}")
        await update_upload_status(session, upload_id, "failed")
        raise
    finally:
        await session.close()

@app.task(
    bind=True,
    name="transcribe_segments",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
async def transcribe_segments(self, upload_id: int, file_path: str, segments: list[dict]):
    """
    GPU-транскрипция сегментов, запись JSON и чистка.
    """
    session = AsyncSessionLocal()
    try:
        whisper = get_whisper()
        results: list[dict] = []

        for seg in segments:
            transcription = whisper.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=False,
                segment=seg,
                batch_size=settings.WHISPER_BATCH_SIZE
            )
            text = " ".join([s.text for s in transcription])
            results.append({**seg, "text": text})

        # сохраняем в JSON рядом с оригиналом
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        await update_upload_status(session, upload_id, "completed")
        cleanup_files(file_path, json_path)

    except Exception as e:
        logger.exception(f"Error in transcribe_segments (upload_id={upload_id}): {e}")
        await update_upload_status(session, upload_id, "failed")
        raise
    finally:
        await session.close()

# ---------- Утилиты ----------
@app.task(name="cleanup_old_files")
def cleanup_old_files():
    """
    Удаляет файлы старше FILE_RETENTION_DAYS в UPLOAD_FOLDER и RESULTS_FOLDER.
    """
    now = int(time.time())
    cutoff = now - settings.FILE_RETENTION_DAYS * 24 * 3600

    for dir_path in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER):
        for root, _, files in os.walk(dir_path):
            for fn in files:
                full = os.path.join(root, fn)
                if os.path.getmtime(full) < cutoff:
                    try:
                        os.remove(full)
                        logger.info(f"Removed old file {full}")
                    except Exception:
                        logger.warning(f"Failed to remove {full}", exc_info=True)

def cleanup_files(*paths: str):
    """Мгновенная чистка конкретных файлов (оригинал + JSON)."""
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"Deleted file {p}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"Failed to delete {p}: {e}")