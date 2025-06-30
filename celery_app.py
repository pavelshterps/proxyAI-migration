from celery import Celery
from celery.schedules import crontab

from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.timezone,
    result_expires=3600,
    task_time_limit=600,
    task_soft_time_limit=550,
    task_acks_late=True,
    task_reject_on_worker_lost=True
)

# Разделение очередей
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    "tasks.cleanup_old_files": {"queue": "maintenance"}
}

# Периодическая очистка старых файлов
celery_app.conf.beat_schedule = {
    "cleanup-old-files": {
        "task": "tasks.cleanup_old_files",
        "schedule": crontab(hour=0, minute=0),
    }
}