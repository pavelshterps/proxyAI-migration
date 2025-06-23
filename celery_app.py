# celery_app.py
from celery import Celery
from dotenv import load_dotenv
import os

# Load .env into os.environ
load_dotenv()

# Create the Celery object, *including* our tasks module by name.
# This ensures even plain `import celery_app` picks up tasks.py.
celery_app = Celery(
    "proxyai",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    include=["tasks"],       # <— tell Celery to load tasks.py
)

# Standard Celery config
celery_app.conf.update(
    result_extended=True,
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    enable_utc=True,
    timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
    # optional default routing, but include is the key fix
    task_routes={
        "tasks.transcribe_full": {"queue": "preprocess"},
        "tasks.diarize_task":     {"queue": "preprocess"},
        "tasks.chunk_by_diarization": {"queue": "preprocess"},
        "tasks.merge_results":    {"queue": "preprocess"},
        "tasks.inference_task":   {"queue": "inference"},
    },
)

# (Optional) if you ever split tasks into apps/sub-packages:
# celery_app.autodiscover_tasks(["tasks"])