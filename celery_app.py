# celery_app.py
from celery import Celery
from kombu import Exchange, Queue
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

# Exchanges & Queues
preprocess_ex = Exchange("preprocess", type="direct")
celery_app.conf.task_queues = [
    Queue("preprocess_cpu", exchange=preprocess_ex, routing_key="preprocess.cpu"),
    Queue("preprocess_gpu", exchange=preprocess_ex, routing_key="preprocess.gpu"),
]

celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu", "routing_key": "preprocess.cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu", "routing_key": "preprocess.gpu"},
}

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.CELERY_TIMEZONE,
    result_expires=3600,
    task_time_limit=600,
    task_soft_time_limit=550,
)

if __name__ == "__main__":
    celery_app.start()