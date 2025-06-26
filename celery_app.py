from celery import Celery
from config.settings import settings

# Создаём приложение Celery и сразу указываем, где искать задачи
celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

# Маршрутизация: какие задачи на какие очереди
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Ограничиваем параллелизм и отключаем предварительную загрузку
celery_app.conf.worker_concurrency = int(settings.CPU_CONCURRENCY)
celery_app.conf.worker_prefetch_multiplier = 1