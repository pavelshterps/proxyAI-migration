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

# Инициализируем Celery
app = Celery("proxyai")
app.config_from_object("config.celery")

# Асинхронный движок и сессии
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

def get_whisper() -> WhisperModel:
    if _whisper is None:
        raise RuntimeError("Whisper model is not loaded")
    return _whisper

def get_diarizer() -> DiarizationPipeline:
    if _diarizer is None:
        raise RuntimeError("Diarizer pipeline is not loaded")
    return _diarizer

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    global _whisper, _diarizer

    # 1) Диаризация на CPU
    try:
        _diarizer = DiarizationPipeline.from_pretrained(settings.PYANNOTE_PIPELINE)
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline: {e}")
        raise

    # 2) Whisper на GPU или fallback на CPU, локальный quantized model path
    model_path = settings.WHISPER_MODEL_PATH
    whisper_init_kwargs = {
        "device": settings.WHISPER_DEVICE,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
    }

    try:
        _whisper = WhisperModel(model_path, **whisper_init_kwargs)
        logger.info(f"✅ Loaded Whisper model from `{model_path}` on {settings.WHISPER_DEVICE}")
    except RuntimeError as e:
        logger.error(f"❌ Failed to load Whisper model on {settings.WHISPER_DEVICE}: {e}")
        if settings.WHISPER_DEVICE != "cpu":
            # попытка падения на CPU
            whisper_init_kwargs["device"] = "cpu"
            _whisper = WhisperModel(model_path, **whisper_init_kwargs)
            logger.info(f"⚠️ Fallback: loaded Whisper model on CPU")
        else:
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
    Шаги:
      1) обновить статус → processing
      2) диаризация
      3) транскрипция сегментов
      4) сохранение JSON
      5) обновить статус → completed/failed, удалить файлы
    """
    session = AsyncSessionLocal()
    try:
        await update_upload_status(session, upload_id, "processing")

        # 1) Диаризация
        diarizer = get_diarizer()
        diarization = diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # 2) Транскрипция каждого сегмента
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

        # 3) Сохраняем в JSON
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

        # 4) Завершаем
        await update_upload_status(session, upload_id, "completed")
        cleanup_files(file_path, json_path)

    except Exception as e:
        logger.exception(f"Error in process_audio (upload_id={upload_id}): {e}")
        await update_upload_status(session, upload_id, "failed")
        raise
    finally:
        await session.close()

def cleanup_files(*paths: str):
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"Deleted file {p}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"Failed to delete {p}: {e}")