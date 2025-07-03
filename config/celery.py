# config/celery.py

from config.settings import settings
from kombu import Queue, Exchange

# Broker и бэкенд
broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

# Сериализация
task_serializer = "json"
accept_content = ["json"]

# Таймзона
timezone = settings.CELERY_TIMEZONE
enable_utc = True

# Определяем две очереди
task_queues = (
    Queue("preprocess_cpu", Exchange("preprocess_cpu"), routing_key="preprocess_cpu"),
    Queue("preprocess_gpu", Exchange("preprocess_gpu"), routing_key="preprocess_gpu"),
)

# Маршрутизация задач
task_routes = {
    # транскрипция идёт на GPU
    "tasks.transcribe_segments": {
        "queue": "preprocess_gpu",
        "routing_key": "preprocess_gpu",
    },
    # диаризация идёт на CPU
    "tasks.diarize_full": {
        "queue": "preprocess_cpu",
        "routing_key": "preprocess_cpu",
    },
}

# (опционально) периодические задачи
beat_schedule = {}