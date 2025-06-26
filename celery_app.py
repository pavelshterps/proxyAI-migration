from celery import Celery
from config.settings import settings

celery_app = Celery(
    __name__,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    timezone=settings.CELERY_TIMEZONE,
    task_routes={
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    },
    task_default_queue="preprocess_cpu",
)