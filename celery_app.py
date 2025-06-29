from celery import Celery
from config.settings import settings

# Create the Celery application
celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],  # ensures tasks.py is imported and its @shared_task definitions registered
)

# Route tasks to the correct queues
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Use JSON serialization and your configured timezone
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.TIMEZONE,
)

# (Optional) further tuning:
# celery_app.conf.result_expires = 3600        # seconds to keep results
# celery_app.conf.task_time_limit = 600        # hard time limit per task
# celery_app.conf.task_soft_time_limit = 550   # soft time limit per task

if __name__ == "__main__":
    # Allows: python celery_app.py worker --loglevel=info
    celery_app.start()