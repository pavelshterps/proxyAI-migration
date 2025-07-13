# celery_app.py

from celery import Celery, signals
from config.settings import settings
from tasks import get_whisper_model, get_diarizer

app = Celery("proxyai")
app.config_from_object("config.celery")

@signals.worker_process_init.connect
def preload_models(**kwargs):
    """
    Прогрев моделей при старте воркера:
      • CPU-воркеры (device='cpu') — PyAnnote-диаризер
      • GPU-воркеры — WhisperModel
    """
    device = settings.WHISPER_DEVICE.lower()
    if device == "cpu":
        get_diarizer()
    else:
        get_whisper_model()