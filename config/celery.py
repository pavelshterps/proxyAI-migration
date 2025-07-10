from config.settings import settings
from kombu import Queue, Exchange

broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

task_serializer = "json"
accept_content = ["json"]

timezone = settings.CELERY_TIMEZONE
enable_utc = True

# Очереди
task_queues = (
    Queue("preprocess_cpu", Exchange("preprocess_cpu"), routing_key="preprocess_cpu"),
    Queue("preprocess_gpu", Exchange("preprocess_gpu"), routing_key="preprocess_gpu"),
)

# Правильная маршрутизация
task_routes = {
    "tasks.transcribe_segments": {
        "queue": "preprocess_cpu",
        "routing_key": "preprocess_cpu",
    },
    "tasks.diarize_full": {
        "queue": "preprocess_gpu",
        "routing_key": "preprocess_gpu",
    },
}

task_acks_late = True
task_reject_on_worker_lost = True
worker_prefetch_multiplier = 1
task_time_limit = 600
task_soft_time_limit = 550

beat_schedule = {
    # сюда ваши периодические задачи
}