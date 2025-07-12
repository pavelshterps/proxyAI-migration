from celery import Celery

from config.settings import settings

app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# единый JSON-сериализатор, таймзона, и т.п.
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    enable_utc=True,
    timezone=settings.TIMEZONE,
)