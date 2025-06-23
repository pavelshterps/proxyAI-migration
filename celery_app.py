from celery import Celery
from dotenv import load_dotenv
import os

# Загружаем переменные окружения из .env
load_dotenv()

# Создаём объект Celery и явно включаем модуль tasks.py
celery_app = Celery(
    "proxyai",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    include=["tasks"],  # явно импортируем tasks.py для регистрации всех задач
)

# Общая конфигурация Celery
celery_app.conf.update(
    result_extended=True,
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    enable_utc=True,
    timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
)

# Явные маршруты задач по очередям
celery_app.conf.task_routes = {
    "tasks.transcribe_full":       {"queue": "preprocess"},
    "tasks.diarize_task":          {"queue": "preprocess"},
    "tasks.chunk_by_diarization":  {"queue": "preprocess"},
    "tasks.merge_results":         {"queue": "preprocess"},
    "tasks.inference_task":        {"queue": "inference"},
}