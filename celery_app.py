# celery_app.py

from config.settings import settings
from celery import Celery

# Инициализируем Celery без include/imports
app = Celery(
    __name__,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Автоматически находим задачи в модуле tasks
app.autodiscover_tasks(["tasks"])
# Настраиваем маршрутизацию по очередям
app.conf.task_routes = {
    # полный файл для диаризации идёт в очередь preprocess_cpu
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    # чанки для транскрипции — в очередь preprocess_gpu
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Опционально можно задать дефолтную очередь
app.conf.task_default_queue = "preprocess_cpu"
app.conf.task_default_exchange = "preprocess_cpu"
app.conf.task_default_routing_key = "preprocess_cpu"