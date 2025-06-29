# celery_app.py

from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],  # гарантируем, что tasks.py подхватится
)

# маршрутизация задач по очередям
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# JSON-сериализация, таймауты, timezones
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.TIMEZONE,
    result_expires=3600,         # час хранения результатов
    task_time_limit=600,         # общий таймаут 10 мин
    task_soft_time_limit=550,    # мягкий таймаут ~9мин10сек
)

if __name__ == "__main__":
    celery_app.start()