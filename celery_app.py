import os
from celery import Celery
from celery.signals import worker_process_init
from config.settings import settings
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = Celery(
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 1
app.conf.broker_transport_options = {"visibility_timeout": 3600}

@worker_process_init.connect
def preload_models(**kwargs):
    from tasks import get_whisper_model, get_diarization_pipeline

    try:
        get_whisper_model()
        get_diarization_pipeline()
        logger.info(f"[{datetime.utcnow().isoformat()}] [PRELOAD] models loaded")
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] [PRELOAD] error loading models: {e}", exc_info=True)