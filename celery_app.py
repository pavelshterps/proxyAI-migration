# celery_app.py
import os
import structlog
from celery import Celery
from config.settings import settings

# configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

celery_app = Celery(
    "proxyai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks"]
)

# route tasks
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.timezone,
    result_expires=3600,
    task_time_limit=600,
    task_soft_time_limit=550,
)

# optionally instrument Prometheus metrics
if settings.enable_metrics:
    from prometheus_client import start_http_server, Counter
    TASK_COUNTER = Counter("celery_tasks_total", "Total number of Celery tasks", ["task"])
    start_http_server(8001)  # expose on port 8001

    @celery_app.task_prerun.connect
    def incr_task_counter(sender=None, **kwargs):
        TASK_COUNTER.labels(task=sender.name).inc()

if __name__ == "__main__":
    celery_app.start()