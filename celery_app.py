from celery import Celery, signals
from tasks import get_whisper, get_diarizer

app = Celery("proxyai")
app.config_from_object("config.celery")

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Для Celery < 5.4: перенаправляем на наши функции из tasks.py
    """
    # триггерим загрузку моделей
    get_diarizer()
    get_whisper()