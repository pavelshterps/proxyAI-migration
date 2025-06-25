import os
from celery import Celery

celery_app = Celery(
    "proxyai",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    include=["tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
    enable_utc=True,
)