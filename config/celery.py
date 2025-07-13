from celery import Celery
from config.settings import settings

# Переименовали экземпляр, чтобы не конфликтовал с FastAPI `app`
celery_app = Celery(
    'proxyai',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    timezone=settings.CELERY_TIMEZONE,
    include=['tasks'],
)

# Общая конфигурация Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
)