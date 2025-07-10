from config.settings import settings
from kombu import Queue, Exchange

# Broker and result backend
broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

# Serialization
task_serializer = "json"
accept_content = ["json"]

# Timezone settings
timezone = settings.CELERY_TIMEZONE
enable_utc = True

# Queues
task_queues = (
    Queue("preprocess_cpu", Exchange("preprocess_cpu"), routing_key="preprocess_cpu"),
    Queue("preprocess_gpu", Exchange("preprocess_gpu"), routing_key="preprocess_gpu"),
)

# Route both tasks onto the GPU queue for now
task_routes = {
    "tasks.transcribe_segments": {
        "queue": "preprocess_gpu",
        "routing_key": "preprocess_gpu",
    },
    "tasks.diarize_full": {
        "queue": "preprocess_gpu",
        "routing_key": "preprocess_gpu",
    },
}

# Reliability / resource control
task_acks_late = True
task_reject_on_worker_lost = True
worker_prefetch_multiplier = 1
task_time_limit = 600        # hard limit (10 min)
task_soft_time_limit = 550   # soft limit

# Periodic tasks
beat_schedule = {
    "cleanup_old_uploads": {
        "task": "tasks.cleanup_old_uploads",
        "schedule": 3600.0,  # run hourly
    },
}

# Ensure Beat imports your task definitions
imports = ("tasks",)