import os
from celery import Celery

from config.settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery("proxyai")
celery_app.conf.update(
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1,
    task_default_queue="default",
    task_routes={
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    },
)

# Import tasks so they get registered with Celery
import tasks  # noqa: F401