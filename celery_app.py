from celery import Celery
from config.settings import settings

# Instantiate the Celery app
celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],  # ensures tasks.py is imported
)

# Route specific tasks to the right queues
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Use JSON for serialization and UTC-based scheduling
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.TIMEZONE,
)

# (Optional) tweak any other Celery settings here:
# celery_app.conf.result_expires = 3600
# celery_app.conf.task_time_limit = 300
# celery_app.conf.task_soft_time_limit = 240

if __name__ == "__main__":
    # allows running with `python celery_app.py`
    celery_app.start()