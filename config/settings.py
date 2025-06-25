from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    ALLOWED_ORIGINS: List[str] = ["*"]

    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int
    MAX_FILE_SIZE: int
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str = "wav"

    DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str
    WHISPER_MODEL: str
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str

    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()