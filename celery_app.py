from celery import Celery
from config.settings import settings

# Имя приложения и настройки брокера/бека
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# обновляем конфиг celery (concurrency, timezone и т.п.)
app.conf.update(
    result_backend=settings.CELERY_RESULT_BACKEND,
    timezone=settings.CELERY_TIMEZONE,
    worker_concurrency=settings.CELERY_CONCURRENCY,
)