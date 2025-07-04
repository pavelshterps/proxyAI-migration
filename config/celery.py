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

# Очереди
task_queues = (
    Queue("preprocess_cpu", Exchange("preprocess_cpu"), routing_key="preprocess_cpu"),
    Queue("preprocess_gpu", Exchange("preprocess_gpu"), routing_key="preprocess_gpu"),
)

# Маршрутизация задач
task_routes = {
    "tasks.diarize_full": {
        "queue": "preprocess_cpu",
        "routing_key": "preprocess_cpu",
    },
    "tasks.transcribe_segments": {
        "queue": "preprocess_gpu",
        "routing_key": "preprocess_gpu",
    },
}

# Рекомендованные опции для надёжности и контроля ресурсов
task_acks_late = True                       # подтверждать при окончании задачи
task_reject_on_worker_lost = True           # автозабалансировка при падении воркера
worker_prefetch_multiplier = 1              # минимальный prefetch (по одному таску на воркер)
task_time_limit = 600                       # жёсткий таймаут в секундах (10 минут)
task_soft_time_limit = 550                  # мягкий таймаут, за 50с до жёсткого

# (продолжайте держать здесь любые периодические задачи)
beat_schedule = {}