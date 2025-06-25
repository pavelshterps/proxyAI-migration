import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    ALLOWED_ORIGINS: list[str] = ["*"]

    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    UPLOAD_FOLDER: str = "/tmp/uploads"
    MAX_FILE_SIZE: int = 1073741824

    DEVICE: str
    WHISPER_COMPUTE_TYPE: str
    WHISPER_MODEL: str
    ALIGN_BEAM_SIZE: int
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str

    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"

settings = Settings()