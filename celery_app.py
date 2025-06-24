import os
from celery import Celery
from config.settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery("proxyai")
celery_app.conf.broker_url = CELERY_BROKER_URL
celery_app.conf.result_backend = CELERY_RESULT_BACKEND
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_acks_late = True
celery_app.conf.worker_max_tasks_per_child = 1
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_routes = {
    "tasks.transcribe_full": {"queue": "default"},
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Импортируем module с задачами, чтобы Celery их зарегистрировал
import tasks  # noqa: F401