# celery_app.py
from celery import Celery
from config.settings import settings

# создаём celery-приложение
celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# конфигурируем очередь (для CPU и GPU воркеров мы будем выбирать разные очереди через --queues)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.model_config.env_file,  # TZ берётся из .env (по-умолчанию UTC)
)