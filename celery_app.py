from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Разделение задач по очередям
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Ограничение консьюмеров (CPU-воркер и GPU-воркер берут настройки из .env)
celery_app.conf.update(
    worker_concurrency=settings.CELERY_CONCURRENCY,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)