import os
from celery import Celery

from config.settings import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_CONCURRENCY,
    CELERY_TIMEZONE
)

# Single app but two route sets
app = Celery("proxyai", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=CELERY_TIMEZONE,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_routes={
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_full": {"queue": "preprocess_gpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    },
)

# ensure all tasks modules are loaded
import tasks  # noqa: F401