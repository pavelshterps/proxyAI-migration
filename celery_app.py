import os
from celery import Celery
from celery.schedules import crontab
from config.settings import settings

app = Celery('whisperx',
             broker=settings.CELERY_BROKER_URL,
             backend=settings.CELERY_RESULT_BACKEND)

app.conf.task_routes = {'tasks.transcribe_task': {'queue': 'transcribe'}}
app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 1
app.conf.task_time_limit = 3600
app.conf.beat_schedule = {
    'cleanup-old-uploads': {
        'task': 'tasks.cleanup_files',
        'schedule': crontab(hour=0, minute=0),
    },
}
app.conf.timezone = settings.CELERY_TIMEZONE
