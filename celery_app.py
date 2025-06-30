# celery_app.py

from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks"],  # make sure tasks.py is imported at startup
)

# Route tasks onto their own queues
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Use JSON everywhere, enable UTC+timezone, plus sensible timeouts
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.timezone,
    result_expires=3600,         # 1 hour
    task_time_limit=600,         # 10 minutes hard
    task_soft_time_limit=550,    # 9 minutes 10 sec soft
)

if __name__ == "__main__":
    celery_app.start()