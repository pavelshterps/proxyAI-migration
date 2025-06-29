import structlog
from celery import Celery
from config.settings import settings

# Structured logging
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
    include=["tasks"],
)

# Route tasks to dedicated queues
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Serialization, timeouts, concurrency
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

if __name__ == "__main__":
    celery_app.start()