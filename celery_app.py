import structlog
from celery import Celery

from config.settings import settings

logger = structlog.get_logger()

celery_app = Celery(
    "proxyai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks"],
)

# Маршрутизация задач по очередям
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Общие настройки Celery
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.timezone,
    result_expires=3600,         # хранить результаты 1 час
    task_time_limit=600,         # жесткий таймаут 10 минут
    task_soft_time_limit=550,    # мягкий таймаут 9 минут 10 секунд
)

if __name__ == "__main__":
    celery_app.start()