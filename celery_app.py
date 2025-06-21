import os
from celery import Celery

broker_url = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/1')

celery_app = Celery(
    'proxyai',
    broker=broker_url,
    backend=result_backend,
    include=['tasks']
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    task_acks_late=True,
    task_time_limit=86400,
    task_soft_time_limit=7200,
    worker_send_task_events=True,
)