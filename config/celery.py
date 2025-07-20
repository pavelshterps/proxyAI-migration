from celery import Celery
from celery.schedules import crontab
from kombu import Queue

from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    timezone=settings.CELERY_TIMEZONE,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # четыре очереди: CPU транскод/preview, GPU preview, GPU full, GPU diarize
    task_queues=[
        Queue("transcribe_cpu"),
        Queue("preview_gpu"),
        Queue("transcribe_gpu"),
        Queue("diarize_gpu"),
    ],
    task_routes={
        "tasks.convert_to_wav_and_preview": {"queue": "transcribe_cpu"},
        "tasks.preview_transcribe":         {"queue": "preview_gpu"},
        "tasks.transcribe_segments":        {"queue": "transcribe_gpu"},
        "tasks.diarize_full":               {"queue": "diarize_gpu"},
    },

    broker_transport_options={
        "sentinels": settings.CELERY_SENTINELS,
        "master_name": settings.CELERY_SENTINEL_MASTER_NAME,
        "socket_timeout": settings.CELERY_SENTINEL_SOCKET_TIMEOUT,
        "retry_on_timeout": True,
        "preload_reconnect": True,
    },

    beat_schedule={
        "daily-cleanup-old-files": {
            "task": "tasks.cleanup_old_files",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)