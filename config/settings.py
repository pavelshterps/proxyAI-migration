import os

# FastAPI
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", '["*"]')

# Celery & Redis
CELERY_BROKER_URL     = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
CELERY_CONCURRENCY    = int(os.getenv("CELERY_CONCURRENCY", "1"))
CELERY_TIMEZONE       = os.getenv("CELERY_TIMEZONE", "UTC")

# File upload/retention
UPLOAD_FOLDER        = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
FILE_RETENTION_DAYS  = int(os.getenv("FILE_RETENTION_DAYS", "7"))
MAX_FILE_SIZE        = int(os.getenv("MAX_FILE_SIZE", "1073741824"))
TUS_ENDPOINT         = os.getenv("TUS_ENDPOINT", "http://tusd:1080/files/")
SNIPPET_FORMAT       = os.getenv("SNIPPET_FORMAT", "wav")

# Device mapping
_dev = os.getenv("DEVICE", "cpu").lower()
if _dev in ("gpu","cuda"):
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Whisper
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_MODEL_NAME   = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")

# WhisperX alignment
ALIGN_MODEL_NAME = os.getenv("ALIGN_MODEL_NAME", WHISPER_MODEL_NAME)
ALIGN_BEAM_SIZE  = int(os.getenv("ALIGN_BEAM_SIZE", "5"))

# Speaker diarization
PYANNOTE_PROTOCOL = os.getenv("PYANNOTE_PROTOCOL", "pyannote/speaker-diarization")

# HuggingFace token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Database (optional)
POSTGRES_DB       = os.getenv("POSTGRES_DB", "whisperx")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "whisperx")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "secret")
DATABASE_URL      = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@postgres:5432/{POSTGRES_DB}"
)

# Custom Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")