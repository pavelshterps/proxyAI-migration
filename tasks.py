import os
import json
import logging
import shutil
import tempfile
import datetime
import subprocess
import asyncio
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init
from celery.schedules import crontab
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from crud import update_upload_status

# === Logging setup ===
logger = logging.getLogger("proxyai.tasks")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s [%(task_id)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_context(task_id: str):
    return {"task_id": task_id}

# === Celery application ===
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
# Алиас для CLI (celery -A tasks ...)
celery = app

app.conf.task_default_queue = "preprocess_cpu"
app.conf.beat_schedule = {
    "cleanup-old-files-every-day": {
        "task": "tasks.cleanup_old_files",
        "schedule": crontab(hour=0, minute=0),
    },
}

# === Database (SQLAlchemy) ===
engine = create_async_engine(settings.DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# === Model singletons ===
_whisper_instance = None
_diarizer = None

def get_whisper():
    global _whisper_instance
    if _whisper_instance is None:
        repo = settings.WHISPER_MODEL_PATH
        device = settings.WHISPER_DEVICE
        compute = settings.WHISPER_COMPUTE_TYPE
        batch   = settings.WHISPER_BATCH_SIZE

        init_kwargs = {
            "cache_dir": settings.HUGGINGFACE_CACHE_DIR,
            "device": device,
            "compute_type": compute,
            "batch_size": batch,
            # Разрешаем докачку из интернета при отсутствии локального кеша
            "local_files_only": False,
            "use_auth_token": settings.HUGGINGFACE_TOKEN,
        }

        logger.info(f"Loading WhisperModel '{repo}' on device={device} compute={compute}")
        try:
            _whisper_instance = WhisperModel(repo, **init_kwargs)
        except Exception as e:
            logger.warning(f"WhisperModel init failed ({e}); retrying local-only")
            init_kwargs["local_files_only"] = True
            _whisper_instance = WhisperModel(repo, **init_kwargs)
        logger.info("WhisperModel loaded")
    return _whisper_instance

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.makedirs(settings.DIARIZER_CACHE_DIR, exist_ok=True)
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("Diarizer loaded")
    return _diarizer

# === Warm-up on worker start ===
@worker_process_init.connect
def preload_and_warmup(**kwargs):
    sample = Path(settings.UPLOAD_FOLDER) / "warmup.wav"

    # Разогрев Diarizer
    try:
        get_diarizer()(str(sample))
        logger.info("✅ Diarizer warm-up complete")
    except Exception as e:
        logger.warning(f"Diarizer warm-up failed: {e}")

    # Разогрев WhisperModel
    try:
        get_whisper()
        logger.info("✅ Whisper warm-up complete")
    except Exception as e:
        logger.warning(f"Whisper warm-up failed: {e}")

# === Celery tasks ===
@app.task(bind=True, name="tasks.diarize_full")
def diarize_full(self, upload_id: str):
    ctx = log_context(upload_id)
    logger.info("Starting diarization", extra=ctx)
    try:
        src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)

        diarization = get_diarizer()(str(src))
        with open(out_dir / "diarization.json", "w", encoding="utf-8") as f:
            f.write(diarization.to_json())

        logger.info("Diarization finished", extra=ctx)
        update_status.delay(upload_id, "diarized")
    except Exception:
        logger.exception("Error in diarize_full", extra=ctx)
        update_status.delay(upload_id, "error_diarization")
        raise

@app.task(bind=True, name="tasks.transcribe_segments")
def transcribe_segments(self, upload_id: str):
    ctx = log_context(upload_id)
    logger.info("Starting transcription", extra=ctx)
    try:
        src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        out_dir = Path(settings.RESULTS_FOLDER) / upload_id

        # Загружаем результаты diarize_full
        segments = json.loads((out_dir / "diarization.json").read_text(encoding="utf-8"))
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
                offset=0,
                duration=None
            )
            for s in result["segments"]:
                transcript.append({
                    "start": s.start,
                    "end": s.end,
                    "speaker": speaker,
                    "text": s.text
                })
            os.unlink(tmp.name)

        with open(out_dir / "transcript.json", "w", encoding="utf-8") as f:
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
    logger.info(f"Updating status to '{status}'", extra=ctx)
    try:
        async def _update():
            async with AsyncSessionLocal() as session:
                await update_upload_status(session, upload_id, status)
        asyncio.get_event_loop().run_until_complete(_update())
        logger.info("Status updated", extra=ctx)
    except Exception:
        logger.exception("Error in update_status", extra=ctx)

@app.task(bind=True, name="tasks.cleanup_old_files")
def cleanup_old_files(self):
    ctx = {"task_id": "cleanup"}
    logger.info("Starting cleanup_old_files", extra=ctx)
    try:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=settings.FILE_RETENTION_DAYS)
        for base in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER):
            for item in Path(base).iterdir():
                try:
                    mtime = datetime.datetime.utcfromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        logger.info(f"Removed {item}", extra=ctx)
                except Exception:
                    logger.exception(f"Failed to remove {item}", extra=ctx)
        logger.info("cleanup_old_files completed", extra=ctx)
    except Exception:
        logger.exception("Error in cleanup_old_files", extra=ctx)

# === Helper functions ===
def extract_audio_segment(src_path: str, dst_path: str, start: float, end: float):
    """Вырезает аудио-сегмент [start, end) из src_path в dst_path."""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ss", str(start), "-to", str(end),
        "-c", "copy", dst_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)