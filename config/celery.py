from config.settings import settings
from kombu import Queue, Exchange
from celery import Celery

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
    "tasks.transcribe_segments": {
        "queue": "preprocess_gpu",      # транскрипция теперь на GPU
        "routing_key": "preprocess_gpu",
    },
    "tasks.diarize_full": {
        "queue": "preprocess_gpu",
        "routing_key": "preprocess_gpu",
    },
}

# Рекомендованные опции для надёжности и контроля ресурсов
task_acks_late = True                       # подтверждать при окончании задачи
task_reject_on_worker_lost = True           # автозабалансировка при падении воркера
worker_prefetch_multiplier = 1              # минимальный prefetch
task_time_limit = 600                       # жёсткий таймаут (10 минут)
task_soft_time_limit = 550                  # мягкий таймаут, за 50с до жёсткого

# Периодические задачи
beat_schedule = {
    "cleanup_old_uploads": {
        "task": "tasks.cleanup_old_uploads",
        "schedule": 3600.0,  # ежечасно
    },
}

# ─── Единый экземпляр Celery ────────────────────────────────────────────────────
app = Celery(
    "proxyai",
    broker=broker_url,
    backend=result_backend,
)

# Применяем сконфигурированные параметры
app.conf.update(
    task_serializer=task_serializer,
    accept_content=accept_content,
    timezone=timezone,
    enable_utc=enable_utc,
    task_queues=task_queues,
    task_routes=task_routes,
    task_acks_late=task_acks_late,
    task_reject_on_worker_lost=task_reject_on_worker_lost,
    worker_prefetch_multiplier=worker_prefetch_multiplier,
    task_time_limit=task_time_limit,
    task_soft_time_limit=task_soft_time_limit,
    beat_schedule=beat_schedule,
)