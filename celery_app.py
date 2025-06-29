from celery import Celery
from kombu import Queue
from config.settings import settings

celery_app = Celery(
    settings.app_name,
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Define two dedicated queues
celery_app.conf.task_queues = (
    Queue("preprocess_cpu", routing_key="preprocess_cpu"),
    Queue("preprocess_gpu", routing_key="preprocess_gpu"),
)

# Route each task by name
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Recycle workers every 100 tasks, enforce time limits
celery_app.conf.worker_max_tasks_per_child = 100
celery_app.conf.task_time_limit = 600            # hard limit: 10m
celery_app.conf.task_soft_time_limit = 300       # soft limit: 5m
celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1