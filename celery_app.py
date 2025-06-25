from celery import Celery
from config.settings import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_TIMEZONE
)

app = Celery(
    "proxyai",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

app.conf.update(
    timezone=CELERY_TIMEZONE,
    task_routes={
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    },
    task_ignore_result=False,
)