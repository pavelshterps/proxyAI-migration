from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],     # ensure tasks.py is loaded
    imports=["tasks"],     # explicitly import tasks module
)

# Route each task onto its own queue
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
    timezone=settings.TIMEZONE,
    result_expires=3600,         # 1 hour
    task_time_limit=600,         # 10 minutes hard
    task_soft_time_limit=550,    # 9m10s soft
)

if __name__ == "__main__":
    celery_app.start()