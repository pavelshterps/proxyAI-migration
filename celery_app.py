import logging
from celery import Celery
from config.settings import settings

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

# Route tasks to queues
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Concurrency settings
celery_app.conf.worker_concurrency = int(settings.CPU_CONCURRENCY)
celery_app.conf.worker_prefetch_multiplier = 1

# Preload Whisper model on GPU workers
try:
    import torch
    if torch.cuda.is_available():
        logger.info("CUDA available, preloading WhisperModel...")
        from tasks import get_whisper_model
        get_whisper_model()
        logger.info("WhisperModel preloaded")
except Exception as e:
    logger.warning(f"Failed to preload WhisperModel: {e}")