from celery import Celery
from config.settings import settings

app = Celery(
    "proxyai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Task routing: CPU‐bound vs GPU‐bound
app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}
app.conf.task_default_queue = "default"
app.conf.worker_prefetch_multiplier = 1  # prevent overscheduling
app.conf.task_acks_late = True

# Let Celery auto‐discover tasks.py
app.autodiscover_tasks(["tasks"])