# celery_app.py

from celery import Celery, signals
from config.settings import settings
from tasks import get_whisper_model, get_diarizer

app = Celery("proxyai")
# читаем всю конфигурацию из config/celery.py
app.config_from_object("config.celery")

@signals.worker_process_init.connect
def preload_models(**kwargs):
    """
    При старте каждого процесса-воркера:
    - CPU-воркеры прогревают PyAnnote-диаризер
    - GPU-воркеры прогревают WhisperModel
    """
    device = settings.WHISPER_DEVICE.lower()
    if device == "cpu":
        get_diarizer()
    else:
        get_whisper_model()