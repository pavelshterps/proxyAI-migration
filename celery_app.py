import os
import sys
import logging
from datetime import datetime

from celery import Celery
from celery.signals import worker_process_init
from celery.schedules import crontab
from kombu import Queue

from config.settings import settings

# Добавляем корень приложения в PYTHONPATH, чтобы tasks всегда находился
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Logger setup ---
logger = logging.getLogger("celery_app")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Создаём Celery
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"],
)

# Общая конфигурация
app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={
        "sentinels": settings.CELERY_SENTINELS,
        "master_name": settings.CELERY_SENTINEL_MASTER_NAME,
        "socket_timeout": settings.CELERY_SENTINEL_SOCKET_TIMEOUT,
        "retry_on_timeout": True,
        "preload_reconnect": True,
        "role": "master",
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.CELERY_TIMEZONE,
    task_queues=[
        Queue("transcribe_cpu"),
        Queue("transcribe_gpu"),
        Queue("diarize_gpu"),
        Queue("webhooks"),  # очередь для вебхуков
    ],
    task_routes={
        "tasks.preview_transcribe": {"queue": "transcribe_gpu"},
        "tasks.transcribe_segments": {"queue": "transcribe_gpu"},
        "tasks.diarize_full": {"queue": "diarize_gpu"},
        # Явная маршрутизация вебхуков
        "tasks.deliver_webhook": {"queue": "webhooks"},
        "deliver_webhook": {"queue": "webhooks"},
    },
    beat_schedule={
        "daily-cleanup-old-files": {
            "task": "tasks.cleanup_old_files",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)


@worker_process_init.connect
def preload_models(**kwargs):
    """
    Аккуратно предзагружаем модели ТОЛЬКО на нужных ролях воркеров.
    Роль берём надёжно: env -> settings -> эвристика по WHISPER_DEVICE.
    """
    import os
    role = (os.getenv("WORKER_ROLE") or getattr(settings, "WORKER_ROLE", "") or "").lower()
    if not role:
        dev = (os.getenv("WHISPER_DEVICE") or getattr(settings, "WHISPER_DEVICE", "") or "").lower()
        if dev.startswith("cuda"):
            role = "gpu"
        elif dev == "cpu":
            role = "cpu"

    from tasks import get_whisper_model, get_diarization_pipeline

    try:
        # Whisper — на транскрайберах/гпу-воркерах
        if role in ("cpu", "gpu", "transcribe", "transcribe_cpu", "transcribe_gpu"):
            get_whisper_model()

        # Диаризация — на диаризационных/гпу-воркерах
        if role in ("diarize", "gpu", "gpu_diarize", "diarize_gpu"):
            get_diarization_pipeline()

        logger.info(f"[{datetime.utcnow().isoformat()}] [PRELOAD] models loaded for role='{role}' "
                    f"(env.WORKER_ROLE='{os.getenv('WORKER_ROLE','')}', settings.WORKER_ROLE='{getattr(settings,'WORKER_ROLE','')}')")
    except Exception as e:
        logger.error(
            f"[{datetime.utcnow().isoformat()}] [PRELOAD] error loading models for role='{role}': {e}",
            exc_info=True,
        )