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
app = Celery("proxyai")
app.config_from_object("config.celery")

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    global _whisper, _diarizer

    # 1) Диаризация
    _diarizer = DiarizationPipeline.from_pretrained(
        settings.PYANNOTE_PIPELINE,
        use_auth_token=settings.HUGGINGFACE_TOKEN,
        cache_dir="/hf_cache"
    )
    logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")

    # 2) Whisper (локально, FP16)
    whisper_opts = {
        "device": settings.WHISPER_DEVICE,
        "device_index": 0,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
        "cpu_threads": settings.CPU_CONCURRENCY if settings.WHISPER_DEVICE == "cpu" else 0,
        "num_workers": settings.GPU_CONCURRENCY if settings.WHISPER_DEVICE.startswith("cuda") else 1,
        "download_root": "/hf_cache",
        "local_files_only": True,
    }
    _whisper = WhisperModel(settings.WHISPER_MODEL_PATH, **whisper_opts)
    logger.info(f"✅ Loaded Whisper model from `{settings.WHISPER_MODEL_PATH}` on `{settings.WHISPER_DEVICE}`")

async def process_audio(self, upload_id: int, file_path: str):
    session: AsyncSession = AsyncSessionLocal()
    try:
        await update_upload_status(session, upload_id, "processing")

        diarization = _diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

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

        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

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