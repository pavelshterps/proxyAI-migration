# tasks.py

import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from celery import Celery, signals
from sqlalchemy.ext.asyncio import AsyncSession
from faster_whisper import WhisperModel
import traceback
import logging

from config.settings import settings
from crud import (
    create_upload_record,
    get_upload_for_user,
    update_upload_status  # we just added this  [oai_citation:3‡raw.githubusercontent.com](https://raw.githubusercontent.com/pavelshterps/proxyAI/main/crud.py)
)

logger = logging.getLogger(__name__)

# Celery app setup
app = Celery("proxyai")
app.conf.broker_url = settings.CELERY_BROKER_URL
app.conf.result_backend = settings.CELERY_RESULT_BACKEND

_whisper = None

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        raw = settings.WHISPER_MODEL_PATH  # e.g. local "/hf_cache/..." or "guillaumekln/faster-whisper-medium"
        # If raw is a local path and exists, use it directly (no download)
        if raw.startswith("/") and Path(raw).exists():
            model_source = raw
            init_kwargs = {
                "device": settings.WHISPER_DEVICE,
                "compute_type": settings.WHISPER_COMPUTE_TYPE,
                "batch_size": settings.WHISPER_BATCH_SIZE,
            }
        else:
            # Otherwise treat as HF repo ID, pull from cache only
            model_source = raw
            init_kwargs = {
                "cache_dir": settings.HUGGINGFACE_CACHE_DIR,
                "local_files_only": True,
                "device": settings.WHISPER_DEVICE,
                "compute_type": settings.WHISPER_COMPUTE_TYPE,
                "batch_size": settings.WHISPER_BATCH_SIZE,
            }

        logger.info(
            f"loading whisper model          "
            f"model={model_source} "
            f"cache_dir={init_kwargs.get('cache_dir')} "
            f"compute_type={init_kwargs.get('compute_type')} "
            f"device={init_kwargs.get('device')}"
        )

        # Passing a local path here lets faster_whisper skip any HF download  [oai_citation:4‡knowledge.zhaoweiguo.com](https://knowledge.zhaoweiguo.com/build/html/ai/llms/huggingfaces/lib_python?utm_source=chatgpt.com)
        _whisper = WhisperModel(model_source, **init_kwargs)
    return _whisper

@signals.worker_ready.connect
def preload_and_warmup(**kwargs):
    """Preload the model on worker startup to reduce latency."""
    try:
        get_whisper()
    except Exception:
        logger.exception("Failed to preload Whisper model")
        raise

@app.task(name="process_upload")
async def process_upload(upload_id: str):
    """Top-level task: diarize entire file (CPU), then split & transcribe (GPU)."""
    # 1) mark as processing
    async with AsyncSession(settings.DB_ENGINE) as db:
        await update_upload_status(db, upload_id, "processing")

    # 2) fetch upload record and file path
    async with AsyncSession(settings.DB_ENGINE) as db:
        record = await get_upload_for_user(db, None, upload_id)
        filepath = record.path

    # 3) CPU-bound diarization phase
    try:
        speakers = diarize_full(filepath)  # returns list of (start, end, speaker_id)
        logger.info(f"Diarization complete, {len(speakers)} segments")
    except Exception:
        logger.exception("Diarization failed")
        async with AsyncSession(settings.DB_ENGINE) as db:
            await update_upload_status(db, upload_id, "failed")
        return

    # 4) GPU-bound transcription phase
    transcriptions = []
    whisper = get_whisper()
    for start, end, speaker in speakers:
        try:
            result = whisper.transcribe_segment(filepath, start, end)
            text = result.text
            transcriptions.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text
            })
        except Exception:
            logger.exception(f"Transcription failed for segment {start}-{end}")
            # continue with others

    # 5) write JSON output
    out_path = Path(filepath).with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"segments": transcriptions}, f, ensure_ascii=False, indent=2)

    # 6) mark as completed
    async with AsyncSession(settings.DB_ENGINE) as db:
        await update_upload_status(db, upload_id, "completed")

def cleanup_all_old_files():
    """Periodically delete temp/uploads older than retention period."""
    root = Path(settings.UPLOAD_DIR)
    cutoff = datetime.utcnow() - timedelta(days=settings.RETENTION_DAYS)
    for file in root.glob("**/*"):
        mtime = datetime.utcfromtimestamp(file.stat().st_mtime)
        if mtime < cutoff:
            try:
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()
                logger.info(f"Removed old file {file}")
            except Exception:
                logger.exception(f"Error removing {file}")