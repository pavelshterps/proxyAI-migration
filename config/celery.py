from config.settings import settings
from kombu import Queue, Exchange
from celery import Celery

# Брокер и бэкенд
broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

# Создаём единый экземпляр Celery и подключаем модуль tasks
app = Celery(
    "proxyai",
    broker=broker_url,
    backend=result_backend,
    include=["tasks"],
)

# Сериализация
app.conf.task_serializer = "json"
app.conf.accept_content = ["json"]

# Таймзона
app.conf.timezone = settings.CELERY_TIMEZONE
app.conf.enable_utc = True

# Определение очередей
app.conf.task_queues = (
    Queue("preprocess_cpu", Exchange("preprocess_cpu"), routing_key="preprocess_cpu"),
    Queue("preprocess_gpu", Exchange("preprocess_gpu"), routing_key="preprocess_gpu"),
)

# Маршрутизация задач
app.conf.task_routes = {
    "tasks.transcribe_segments": {"queue": "preprocess_cpu", "routing_key": "preprocess_cpu"},
    "tasks.diarize_full":       {"queue": "preprocess_gpu", "routing_key": "preprocess_gpu"},
}

# Надёжность и контроль ресурсов
app.conf.task_acks_late               = True
app.conf.task_reject_on_worker_lost   = True
app.conf.worker_prefetch_multiplier   = 1
app.conf.task_time_limit              = 600
app.conf.task_soft_time_limit         = 550

# Периодические задачи
app.conf.beat_schedule = {
    "cleanup_old_uploads": {
        "task":    "tasks.cleanup_old_uploads",
        "schedule": 3600.0,  # ежечасно
    },
}