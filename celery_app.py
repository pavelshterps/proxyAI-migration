from celery import Celery
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Create the Celery application, including our tasks module
celery_app = Celery(
    "proxyai",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    include=["tasks"],  # ensures tasks.py is imported
)

# Basic Celery configuration
celery_app.conf.update(
    result_extended=True,
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    enable_utc=True,
    timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
)

# Route only the transcribe_full task onto the preprocess queue
celery_app.conf.task_routes = {
    "tasks.transcribe_full": {"queue": "preprocess"},
}