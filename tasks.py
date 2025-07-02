import os
import json
import shutil
import tempfile
import uuid
import datetime
import logging
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from crud import update_upload_status

# === Logging ===
logger = logging.getLogger("proxyai.tasks")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s [%(task_id)s] %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_context(task_id: str):
    return {"task_id": task_id}

# === Celery ===
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
# Для CLI: alias под имя по умолчанию
celery = app

app.conf.task_default_queue = "preprocess_cpu"
app.conf.beat_schedule = {
    "cleanup-old-files-every-day": {
        "task": "tasks.cleanup_old_files",
        "schedule": crontab(hour=0, minute=0),
    },
}

# === SQLAlchemy ===
engine = create_async_engine(settings.DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# === Pyannote диаризация (CPU) ===
pyannote_pipeline = Pipeline.from_pretrained(settings.PYANNO_PIPELINE)

def extract_audio_segment(src_path: str, dst_path: str, start: float, end: float):
    """Вырезает отрезок audio [start, end) из src_path и сохраняет в dst_path."""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ss", str(start), "-to", str(end),
        "-c", "copy", dst_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@app.task(bind=True, name="tasks.diarize_full")
def diarize_full(self, upload_id: str):
    ctx = log_context(upload_id)
    logger.info("Starting diarization", extra=ctx)
    try:
        src = Path(settings.UPLOAD_FOLDER) / upload_id
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")
        audio = {"uri": upload_id, "audio": str(src)}
        diarization = pyannote_pipeline(audio)
        out = Path(settings.RESULTS_FOLDER) / upload_id
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "diarization.json", "w") as f:
            f.write(diarization.to_json())
        logger.info("Diarization finished", extra=ctx)
        update_status.delay(upload_id, "diarized")
    except Exception:
        logger.exception("Error in diarize_full", extra=ctx)
        update_status.delay(upload_id, "error_diarization")
        raise

# === Whisper предзагрузка (GPU) ===
_whisper_instance = None

def get_whisper():
    global _whisper_instance
    if _whisper_instance:
        return _whisper_instance

    raw = settings.WHISPER_MODEL_PATH
    # Преобразование локального кэша в repo_id
    if raw.startswith("/") and Path(raw).exists():
        name = Path(raw).name
        if name.startswith("models--"):
            repo = name[len("models--"):].replace("--", "/")
        else:
            repo = raw
        model_id = repo
    else:
        model_id = raw

    init_kwargs = {
        "cache_dir": settings.HUGGINGFACE_CACHE_DIR,
        "device": settings.WHISPER_DEVICE,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
        "batch_size": settings.WHISPER_BATCH_SIZE,
        "local_files_only": True,
    }
    logger.info(f"Loading Whisper model `{model_id}`", extra={"task_id": "whisper_init"})
    _whisper_instance = WhisperModel(model_id, **init_kwargs)
    logger.info("Whisper model loaded", extra={"task_id": "whisper_init"})
    return _whisper_instance

@app.task(bind=True, name="tasks.transcribe_segments")
def transcribe_segments(self, upload_id: str, correlation_id: str = None):
    ctx = log_context(upload_id)
    logger.info("Starting transcription", extra=ctx)
    try:
        src = Path(settings.UPLOAD_FOLDER) / upload_id
        outdir = Path(settings.RESULTS_FOLDER) / upload_id
        if not outdir.exists() or not (outdir / "diarization.json").exists():
            raise FileNotFoundError("Diarization JSON not found")
        segments = json.loads((outdir / "diarization.json").read_text())

        whisper = get_whisper()
        transcript = []
        for seg in segments:
            start, end, speaker = seg["start"], seg["end"], seg["speaker"]
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            extract_audio_segment(str(src), tmp.name, start, end)
            result = whisper.transcribe(
                tmp.name,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=settings.VAD_PARAMS,
                chunk_silence_threshold=settings.SILENCE_THRESHOLD
            )
            for s in result["segments"]:
                transcript.append({
                    "start": s.start,
                    "end": s.end,
                    "speaker": speaker,
                    "text": s.text
                })
            os.unlink(tmp.name)

        with open(outdir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        logger.info("Transcription finished", extra=ctx)
        update_status.delay(upload_id, "transcribed")
    except Exception:
        logger.exception("Error in transcribe_segments", extra=ctx)
        update_status.delay(upload_id, "error_transcription")
        raise

@app.task(bind=True, name="tasks.update_status")
def update_status(self, upload_id: str, status: str):
    ctx = log_context(upload_id)
    logger.info(f"Updating status to `{status}`", extra=ctx)
    try:
        async def _update():
            async with AsyncSessionLocal() as session:
                await update_upload_status(session, upload_id, status)
        import asyncio
        asyncio.get_event_loop().run_until_complete(_update())
        logger.info("Status updated", extra=ctx)
    except Exception:
        logger.exception("Error in update_status", extra=ctx)

@app.task(bind=True, name="tasks.cleanup_old_files")
def cleanup_old_files(self):
    logger.info("Starting cleanup_old_files", extra={"task_id": "cleanup"})
    try:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=settings.FILE_RETENTION_DAYS)
        for base in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER):
            folder = Path(base)
            for item in folder.iterdir():
                try:
                    mtime = datetime.datetime.utcfromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff:
                        if item.is_dir():
                            shutil.rmtree(item)
                            logger.info(f"Removed directory {item}", extra={"task_id": "cleanup"})
                        else:
                            item.unlink()
                            logger.info(f"Removed file {item}", extra={"task_id": "cleanup"})
                except Exception:
                    logger.exception(f"Failed to remove {item}", extra={"task_id": "cleanup"})
        logger.info("cleanup_old_files completed", extra={"task_id": "cleanup"})
    except Exception:
        logger.exception("Error in cleanup_old_files", extra={"task_id": "cleanup"})
        raise

@worker_ready.connect
def preload_and_warmup(**kwargs):
    try:
        get_whisper()
    except Exception:
        logger.exception("Error in preload_and_warmup", extra={"task_id": "preload"})