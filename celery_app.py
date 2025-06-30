# celery_app.py

from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks"],
)

# Route tasks onto their queues
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# JSON everywhere, sensible timeouts, UTC
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.timezone,
    result_expires=3600,         # 1 hour
    task_time_limit=600,         # 10 minutes hard
    task_soft_time_limit=550,    # 9m10s soft
)

if __name__ == "__main__":
    celery_app.start()