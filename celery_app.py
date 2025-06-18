import os
from celery import Celery
from config.settings import settings

# Настройка пути для celery, чтобы tasks корректно импортировались
os.environ.setdefault("C_FORCE_ROOT", "true")
os.environ.setdefault("CELERY_CONFIG_MODULE", "config.settings")

celery = Celery(
    "whisperx",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],  # Явно указываем tasks для регистрации всех задач
)

# Опционально — настройки из объекта settings
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=True,
    worker_max_tasks_per_child=10,  # Для безопасности
    broker_connection_retry_on_startup=True,
)

# Необязательно: но можно оставить ручной импорт для совместимости
# import tasks