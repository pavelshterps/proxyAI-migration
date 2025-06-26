# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    UPLOAD_FOLDER: str                   = "/tmp/uploads"
    CELERY_BROKER_URL: str               = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str           = "redis://redis:6379/1"
    HF_CACHE: str                        = "/hf_cache"
    PYANNOTE_MODEL: str                  = "pyannote/speaker-diarization"
    WHISPER_MODEL: str                   = "Systran/faster-whisper-large-v3"
    WHISPER_COMPUTE_TYPE: str            = "float16"
    CHUNK_LENGTH_S: int                  = 30
    WORKER_CPU_CONCURRENCY: int          = 4
    WORKER_GPU_CONCURRENCY: int          = 1
    CELERY_CONCURRENCY: int              = 4
    CELERY_TIMEZONE: str                 = "UTC"

    model_config = SettingsConfigDict(
        env_file = ".env",
        extra    = "ignore",
    )

settings = Settings()