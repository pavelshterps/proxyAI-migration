from celery import Celery
from config.settings import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_TIMEZONE,
    CELERY_CONCURRENCY,
)

# Создаем Celery-приложение с именем "proxyai"
app = Celery(
    "proxyai",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# Настройки Celery
app.conf.update(
    timezone=CELERY_TIMEZONE,
    enable_utc=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_concurrency=int(CELERY_CONCURRENCY),
)

# Маршрутизация задач по очередям
app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    "tasks.transcribe_full": {"queue": "preprocess_cpu"},
}