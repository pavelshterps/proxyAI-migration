# celery_app.py

from config.settings import settings
from celery import Celery

# Instantiate the app as “app” (not celery_app), include our tasks module
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

# Route the two entry-point tasks onto separate queues
app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Default to the CPU queue if nothing else is specified
app.conf.task_default_queue = "preprocess_cpu"
app.conf.task_default_exchange = "preprocess_cpu"
app.conf.task_default_routing_key = "preprocess_cpu"