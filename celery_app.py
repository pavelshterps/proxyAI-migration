import logging
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init

from config.settings import settings

# retain warm-up logs
logger = logging.getLogger("celery_app")

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_send_task_events=True,
    task_send_sent_event=True,
    timezone=settings.CELERY_TIMEZONE,
)

# route transcription to GPU queue, diarization & cleanup to CPU/maintenance
celery_app.conf.task_routes = {
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.cleanup_old_files": {"queue": "maintenance"},
}

celery_app.conf.beat_schedule = {
    "cleanup-old-files": {
        "task": "tasks.cleanup_old_files",
        "schedule": crontab(hour=0, minute=0),
    }
}

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    logger.info("preloading models")
    from tasks import get_whisper, get_diarizer

    # load both models into worker memory
    get_whisper()
    get_diarizer()