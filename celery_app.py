import os
from celery import Celery

# читаем настройки из окружения
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")
CELERY_TIMEZONE = os.getenv("CELERY_TIMEZONE", "UTC")

celery_app = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# основные настройки
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=CELERY_TIMEZONE,
)

# автопоиск всех @celery_app.task в модуле tasks.py
celery_app.autodiscover_tasks(["tasks"])