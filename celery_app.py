import os
from celery import Celery

# Настройки из окружения
broker = os.getenv("CELERY_BROKER_URL")
backend = os.getenv("CELERY_RESULT_BACKEND")
tz      = os.getenv("CELERY_TIMEZONE", "UTC")

celery_app = Celery(
    __name__,
    broker=broker,
    backend=backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=tz,
)

# Autodiscover tasks in tasks.py without circular import
celery_app.autodiscover_tasks(["tasks"])