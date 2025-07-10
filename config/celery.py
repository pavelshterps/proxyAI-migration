from celery import Celery
from kombu import Queue, Exchange
from config.settings import settings

# Broker и бэкенд
broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

# Инициализируем единый экземпляр Celery
app = Celery(
    "proxyai",
    broker=broker_url,
    backend=result_backend,
)

# Общие настройки
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=True,

    # Очереди
    task_queues=(
        Queue("preprocess_cpu", Exchange("preprocess_cpu"), routing_key="preprocess_cpu"),
        Queue("preprocess_gpu", Exchange("preprocess_gpu"), routing_key="preprocess_gpu"),
    ),

    # Все задачи по умолчанию в GPU-очередь
    task_default_queue="preprocess_gpu",

    # Маршрутизация: отправляем обе задачи в preprocess_gpu
    task_routes={
        "tasks.transcribe_segments": {"queue": "preprocess_gpu", "routing_key": "preprocess_gpu"},
        "tasks.diarize_full":       {"queue": "preprocess_gpu", "routing_key": "preprocess_gpu"},
    },

    # Надёжность и контроль ресурсов
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=600,
    task_soft_time_limit=550,

    # Периодические задачи
    beat_schedule={
        "cleanup_old_uploads": {
            "task": "tasks.cleanup_old_uploads",
            "schedule": 3600.0,  # раз в час
        },
    },
)

# Автоматически подхватываем задачи из модуля tasks.py
app.autodiscover_tasks(["tasks"])