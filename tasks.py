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

app = Celery("proxyai")
app.config_from_object("config.celery")

# SQLAlchemy engine & session factory
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker( engine, expire_on_commit=False, class_=AsyncSession )

_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Preload models:
      1) pyannote diarization (CPU)
      2) WhisperModel:
         - float16 on CUDA if requested
         - else default→float32 on CPU
    """
    global _whisper, _diarizer

    # 1) Diarization
    try:
        _diarizer = DiarizationPipeline.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline: {e}", exc_info=True)
        raise

    # 2) Whisper
    model_path = settings.WHISPER_MODEL_PATH

    # Decide compute_type per device
    if settings.WHISPER_DEVICE.startswith("cuda"):
        compute = settings.WHISPER_COMPUTE_TYPE  # e.g. "float16"
    else:
        compute = "default"  # force float32 on CPU

    whisper_kwargs = {
        "device": settings.WHISPER_DEVICE,
        "device_index": 0,
        "compute_type": compute,
    }
    # local cache if set
    if settings.HUGGINGFACE_CACHE_DIR:
        whisper_kwargs["cache_dir"] = settings.HUGGINGFACE_CACHE_DIR

    try:
        _whisper = WhisperModel(model_path, **whisper_kwargs)
        logger.info(
            f"✅ Loaded Whisper model from `{model_path}` "
            f"on `{settings.WHISPER_DEVICE}` as `{compute}`"
        )
    except Exception as e:
        logger.error(f"❌ Whisper load failed ({compute}): {e}", exc_info=True)
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
    1) diarize
    2) transcribe segments
    3) save JSON & update status
    """
    session: AsyncSession = AsyncSessionLocal()
    try:
        await update_upload_status(session, upload_id, "processing")

        # diarization
        diarization = _diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # transcription
        results = []
        for seg in segments:
            out = _whisper.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=False,
                segment=seg
            )
            text = " ".join(chunk.text for chunk in out)
            results.append({**seg, "text": text})

        # write JSON
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        await update_upload_status(session, upload_id, "completed")
        cleanup_files(file_path, json_path)

    except Exception:
        logger.exception(f"Error in process_audio (upload_id={upload_id})")
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