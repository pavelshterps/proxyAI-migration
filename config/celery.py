from celery.schedules import crontab
from kombu import Queue
from config.settings import settings

broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND
timezone = settings.CELERY_TIMEZONE

task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"

imports = ["tasks"]

task_queues = [
    Queue("transcribe_cpu"),
    Queue("transcribe_gpu"),
    Queue("diarize_gpu"),
]
task_routes = {
    "tasks.preview_transcribe":  {"queue": "transcribe_gpu"},
    "tasks.transcribe_segments": {"queue": "transcribe_gpu"},
    "tasks.diarize_full":        {"queue": "diarize_gpu"},
}

broker_transport_options = {
    "sentinels": settings.CELERY_SENTINELS,
    "master_name": settings.CELERY_SENTINEL_MASTER_NAME,
    "socket_timeout": settings.CELERY_SENTINEL_SOCKET_TIMEOUT,
    "retry_on_timeout": True,
    "preload_reconnect": True,
}

beat_schedule = {
    "daily-cleanup-old-files": {
        "task": "tasks.cleanup_old_files",
        "schedule": crontab(hour=3, minute=0),
    },
}