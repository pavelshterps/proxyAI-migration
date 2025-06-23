import os

# FastAPI
FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))
API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

# CORS
ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", '["*"]')

# Celery & Redis
CELERY_BROKER_URL: str     = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
CELERY_CONCURRENCY: int    = int(os.getenv("CELERY_CONCURRENCY", "1"))
CELERY_TIMEZONE: str       = os.getenv("CELERY_TIMEZONE", "UTC")

# Uploads
UPLOAD_FOLDER: str       = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
FILE_RETENTION_DAYS: int = int(os.getenv("FILE_RETENTION_DAYS", "7"))
MAX_FILE_SIZE: int       = int(os.getenv("MAX_FILE_SIZE", str(1024 * 1024 * 1024)))
MAX_FILE_SIZE_MB: int    = MAX_FILE_SIZE // (1024 * 1024)

TUS_ENDPOINT: str   = os.getenv("TUS_ENDPOINT", "http://tusd:1080/files/")
SNIPPET_FORMAT: str = os.getenv("SNIPPET_FORMAT", "wav")

# Hugging Face & caching
HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
HF_CACHE_DIR: str      = os.getenv("HF_CACHE_DIR", "/hf_cache")

# Model parameters
WHISPER_MODEL_NAME: str   = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
ALIGN_MODEL_NAME: str     = os.getenv("ALIGN_MODEL_NAME", "whisper-large")
ALIGN_BEAM_SIZE: int      = int(os.getenv("ALIGN_BEAM_SIZE", "5"))

# Device and compute for Whisper
DEVICE: str               = os.getenv("DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float32")

# Pyannote protocol
PYANNOTE_PROTOCOL: str    = os.getenv("PYANNOTE_PROTOCOL", "pyannote/speaker-diarization")

# Database & Redis URLs
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
REDIS_URL: str    = os.getenv("REDIS_URL", "redis://redis:6379")