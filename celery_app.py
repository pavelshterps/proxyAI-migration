from celery import Celery
from kombu import Exchange, Queue
from config.settings import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_CONCURRENCY,
    CELERY_TIMEZONE,
)

celery_app = Celery(
    'proxyai',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    timezone=CELERY_TIMEZONE,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_concurrency=CELERY_CONCURRENCY,
)

# Две очереди: для CPU-задач и для GPU-задач
celery_app.conf.task_queues = (
    Queue('preprocess_cpu', Exchange('preprocess_cpu'), routing_key='preprocess_cpu'),
    Queue('preprocess_gpu', Exchange('preprocess_gpu'), routing_key='preprocess_gpu'),
)

# Явно указываем, что нужно импортировать модуль tasks
celery_app.conf.task_imports = ('tasks',)

# Импортируем tasks, чтобы все декораторы @celery_app.task сработали
import tasks  # noqa: F401