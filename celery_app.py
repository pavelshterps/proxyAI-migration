from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Регистрируем задачи из модуля tasks.py
celery_app.autodiscover_tasks(["tasks"])

# Настройки: одно воркер-процесс на GPU
celery_app.conf.update(
    task_queues={
        "preprocess_cpu": {"exchange": "preprocess_cpu", "binding_key": "preprocess_cpu"},
        "preprocess_gpu": {"exchange": "preprocess_gpu", "binding_key": "preprocess_gpu"},
    },
    worker_concurrency=settings.CELERY_CONCURRENCY,
    task_default_queue="preprocess_cpu",
)