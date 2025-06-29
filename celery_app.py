from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Core Celery configuration
celery_app.conf.update(
    # Use JSON for task serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Expire task results after 24 hours
    result_expires=24 * 3600,

    # Timezone settings
    timezone="UTC",
    enable_utc=True,

    # Ensure tasks are acknowledged only after execution
    task_acks_late=True,
    worker_prefetch_multiplier=1,

    # Centralized task routing to specific queues
    task_routes={
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    },

    # Optional additional settings (раскомментировать по необходимости):
    # task_time_limit=60 * 60,           # Жесткий таймаут для задач (в секундах)
    # result_backend_transport_options={"visibility_timeout": 3600},
)