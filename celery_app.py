from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_routes={
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=settings.CELERY_TIMEZONE,
)