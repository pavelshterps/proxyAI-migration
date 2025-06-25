import os
from celery import Celery
from config.settings import settings

# заставляем HF-hub кешировать в нашем volume
os.environ.setdefault("HF_HOME", "/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/hf_cache")

celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.timezone = settings.CELERY_TIMEZONE