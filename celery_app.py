# celery_app.py
from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker = settings.CELERY_BROKER_URL,
    backend = settings.CELERY_RESULT_BACKEND,
)

# Маршрутизация задач по очередям
celery_app.conf.task_routes = {
    "tasks.diarize_full":       {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}
celery_app.conf.task_default_queue    = "preprocess_cpu"
celery_app.conf.task_default_exchange = "tasks"
celery_app.conf.task_default_routing_key = "tasks"
celery_app.conf.worker_concurrency    = settings.CELERY_CONCURRENCY
celery_app.conf.timezone              = settings.CELERY_TIMEZONE