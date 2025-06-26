from celery import Celery
from config.settings import settings

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
# route tasks to separate queues
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}
# CPU worker threads & acks
celery_app.conf.worker_concurrency = settings.CELERY_CONCURRENCY
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True