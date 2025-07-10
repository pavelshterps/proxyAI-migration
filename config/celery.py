from celery import Celery
from kombu import Queue, Exchange
from config.settings import settings

# Создаём единственный экземпляр Celery приложения
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Общие настройки Celery
app.conf.update(
    # Сериализация
    task_serializer="json",
    accept_content=["json"],

    # Часовой пояс
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=True,

    # Очередь по умолчанию — GPU
    task_default_queue="preprocess_gpu",

    # Определяем единственную очередь preprocess_gpu
    task_queues=(
        Queue("preprocess_gpu", Exchange("preprocess_gpu"), routing_key="preprocess_gpu"),
    ),

    # Все задачи шлём в preprocess_gpu
    task_routes={
        "tasks.transcribe_segments": {"queue": "preprocess_gpu", "routing_key": "preprocess_gpu"},
        "tasks.diarize_full":       {"queue": "preprocess_gpu", "routing_key": "preprocess_gpu"},
        "tasks.cleanup_old_uploads": {"queue": "preprocess_gpu", "routing_key": "preprocess_gpu"},
    },

    # Подтверждение задач после выполнения
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Минимальный prefetch
    worker_prefetch_multiplier=1,

    # Тайм-ауты
    task_time_limit=600,
    task_soft_time_limit=550,

    # Планировщик периодических задач (Beat)
    beat_schedule={
        "cleanup_old_uploads": {
            "task": "tasks.cleanup_old_uploads",
            "schedule": 3600.0,  # каждую секунду
        },
    },
)