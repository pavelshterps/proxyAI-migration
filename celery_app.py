from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Маршрутизация задач по очередям
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Общие настройки воркера
celery_app.conf.update(
    worker_concurrency=1,  # общее prefetch для стабильности
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    timezone=settings.CELERY_TIMEZONE,
)