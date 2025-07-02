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

# Initialize Celery from our config/celery.py
app = Celery("proxyai")
app.config_from_object("config.celery")

# Async SQLAlchemy engine and session factory
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

# Global singletons
_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Preload models on worker start:
      1) pyannote diarization (CPU)
      2) faster-whisper WhisperModel (FP16 quantized)
    """
    global _whisper, _diarizer

    # 1) load diarization
    try:
        _diarizer = DiarizationPipeline.from_pretrained(settings.PYANNOTE_PIPELINE)
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline: {e}", exc_info=True)
        raise

    # 2) load Whisper (quantized FP16)
    model_path = settings.WHISPER_MODEL_PATH
    whisper_kwargs = {
        "device": settings.WHISPER_DEVICE,             # e.g. "cuda" or "cpu"
        "device_index": 0,                             # first GPU
        "compute_type": settings.WHISPER_COMPUTE_TYPE, # "float16"
        # NB: faster-whisper no longer needs inter_threads/intra_threads here
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
    session: AsyncSession = AsyncSessionLocal()
    try:
        # mark as processing
        await update_upload_status(session, upload_id, "processing")

        # diarize
        diarization = _diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # transcribe each segment
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

        # write JSON
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

        # done
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