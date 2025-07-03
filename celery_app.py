# celery.py

from celery import Celery, signals
from tasks import get_whisper, get_diarizer

app = Celery("proxyai")
# читаем всю конфигурацию из config/celery.py
app.config_from_object("config.celery")

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    При старте каждого воркера сразу инициализируем модели
    (чтобы первый таск не тратил на это время).
    """
    get_diarizer()
    get_whisper()