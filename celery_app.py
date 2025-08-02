import os
import sys
import logging
from datetime import datetime

from celery import Celery
from celery.signals import worker_process_init
from celery.schedules import crontab
from kombu import Queue

from config.settings import settings

# Добавляем корень приложения в PYTHONPATH, чтобы tasks всегда находился
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Logger setup ---
logger = logging.getLogger("celery_app")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Создаём Celery
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

# Общая конфигурация
app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={
        "sentinels": settings.CELERY_SENTINELS,
        "master_name": settings.CELERY_SENTINEL_MASTER_NAME,
        "socket_timeout": settings.CELERY_SENTINEL_SOCKET_TIMEOUT,
        "retry_on_timeout": True,
        "preload_reconnect": True,
        "role": "master",
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.CELERY_TIMEZONE,
    task_queues=[
        Queue("transcribe_cpu"),
        Queue("transcribe_gpu"),
        Queue("diarize_gpu"),
    ],
    task_routes={
        "tasks.preview_transcribe": {"queue": "transcribe_gpu"},
        "tasks.transcribe_segments": {"queue": "transcribe_gpu"},
        "tasks.diarize_full": {"queue": "diarize_gpu"},
    },
    beat_schedule={
        "daily-cleanup-old-files": {
            "task": "tasks.cleanup_old_files",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)


@worker_process_init.connect
def preload_models(**kwargs):
    try:
        from tasks import get_whisper_model, get_diarization_pipeline
        get_whisper_model()
        get_diarization_pipeline()
        logger.info(f"[{datetime.utcnow().isoformat()}] [PRELOAD] models loaded")
    except Exception as e:
        logger.error(
            f"[{datetime.utcnow().isoformat()}] [PRELOAD] error loading models: {e}",
            exc_info=True,
        )