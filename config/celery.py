from celery import Celery
from celery.schedules import crontab
from kombu import Queue
from config.settings import settings

# 1) Сначала создаём приложение
celery_app = Celery(
    'proxyai',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    timezone=settings.CELERY_TIMEZONE,
    include=['tasks'],
)

# 2) Затем настраиваем его
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,

    task_queues=[
        Queue('transcribe_cpu'),
        Queue('transcribe_gpu'),
        Queue('diarize_gpu'),
    ],
    # Теперь у нас единый таск preview_transcribe → transcribe_gpu
    task_routes={
        'tasks.preview_transcribe':    {'queue': 'transcribe_gpu'},
        'tasks.transcribe_segments':   {'queue': 'transcribe_gpu'},
        'tasks.diarize_full':          {'queue': 'diarize_gpu'},
    },
)

# Расписание для очистки старых файлов
celery_app.conf.beat_schedule = {
    'daily-cleanup-old-files': {
        'task': 'tasks.cleanup_old_files',
        'schedule': crontab(hour=3, minute=0),
    },
}