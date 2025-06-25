# celery_app.py

from config.settings import settings
from celery import Celery

# Инициализируем Celery, указывая явно, что нужно импортировать наш модуль tasks
celery_app = Celery(
    __name__,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],       # <- вот это добавлено
)

# Автоматически находим задачи в модуле tasks
celery_app.autodiscover_tasks(["tasks"])  # <- и вот это
celery_app.conf.imports = ["tasks"]
# Настраиваем маршрутизацию по очередям
celery_app.conf.task_routes = {
    # полный файл для диаризации идёт в очередь preprocess_cpu
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    # чанки для транскрипции — в очередь preprocess_gpu
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Опционально можно задать дефолтную очередь
celery_app.conf.task_default_queue = "preprocess_cpu"
celery_app.conf.task_default_exchange = "preprocess_cpu"
celery_app.conf.task_default_routing_key = "preprocess_cpu"