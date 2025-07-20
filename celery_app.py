from celery import Celery, signals
from config.settings import settings

# Создаём Celery и грузим настройки из config/celery.py
app = Celery("proxyai")
app.config_from_object("config.celery")

@signals.worker_process_init.connect
def preload_models(**kwargs):
    """
    Прогрев моделей при старте воркера:
      • CPU-воркеры — инициализируем pyannote-диаризер
      • GPU-воркеры — инициализируем WhisperModel
    """
    device = settings.WHISPER_DEVICE.lower()
    if device == "cpu":
        from tasks import get_clustering_diarizer
        get_clustering_diarizer()
    else:
        from tasks import get_whisper_model
        get_whisper_model()