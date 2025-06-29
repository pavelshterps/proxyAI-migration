from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    result_expires=3600,
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
        "tasks.diarize_full":        {"queue": "preprocess_cpu"},
    },
)