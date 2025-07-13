# config/celery.py

from celery import Celery
from config.settings import settings

celery_app = Celery(
    'proxyai',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    timezone=settings.CELERY_TIMEZONE,
    include=['tasks'],     # автозагрузка вашего модуля с тасками
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,

    # 📌 Прогоним все транскрибирующие таски сразу на GPU-очередь,
    # чтобы gpu-воркер подхватывал их без задержки
    task_routes={
        'tasks.preview_transcribe':  {'queue': 'transcribe_gpu'},
        'tasks.transcribe_segments': {'queue': 'transcribe_gpu'},
    },
)