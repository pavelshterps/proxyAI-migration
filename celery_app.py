import logging
from pathlib import Path

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init

from config.settings import settings

# чтобы логи warm-up не терялись
logger = logging.getLogger("celery_app")

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone=settings.CELERY_TIMEZONE,
    result_expires=3600,
    task_time_limit=600,
    task_soft_time_limit=550,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)

celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    "tasks.cleanup_old_files": {"queue": "maintenance"},
}

celery_app.conf.beat_schedule = {
    "cleanup-old-files": {
        "task": "tasks.cleanup_old_files",
        "schedule": crontab(hour=0, minute=0),
    },
}

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    # предзагрузка моделей
    from tasks import get_whisper, get_diarizer
    whisper = get_whisper()
    diarizer = get_diarizer()

    # warm-up на коротком sample.wav
    sample = Path(__file__).parent / "tests" / "fixtures" / "sample.wav"
    if sample.exists():
        try:
            logger.info("warmup: whisper.transcribe start", path=str(sample))
            whisper.transcribe(
                str(sample),
                offset=0.0,
                duration=2.0,
                language="ru",
                vad_filter=True,
                word_timestamps=False
            )
            logger.info("warmup: whisper.transcribe done")
        except Exception as e:
            logger.warning("warmup: whisper failed", exc_info=e)

        try:
            logger.info("warmup: diarizer start", path=str(sample))
            diarizer(str(sample))
            logger.info("warmup: diarizer done")
        except Exception as e:
            logger.warning("warmup: diarizer failed", exc_info=e)
    else:
        logger.warning("warmup: sample.wav not found, skipping")