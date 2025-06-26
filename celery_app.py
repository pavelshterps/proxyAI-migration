from celery import Celery
from config.settings import settings

# создаём сам Celery-приложение
app = Celery(
    "proxyai",
    broker=settings.BROKER_URL,
    backend=settings.RESULT_BACKEND,
)

# маршрутизация задач по очередям
app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# дополнительные настройки
app.conf.update(
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    timezone="UTC",
    enable_utc=True,
)