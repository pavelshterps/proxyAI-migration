from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str
    FASTAPI_PORT: int
    API_WORKERS: int

    # CORS
    ALLOWED_ORIGINS: List[str]

    # Celery & Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int
    CELERY_TIMEZONE: str

    # Files
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int
    MAX_FILE_SIZE: int
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str

    # Models
    DEVICE: str
    WHISPER_COMPUTE_TYPE: str
    WHISPER_MODEL: str
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str

    # Postgres
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    # URLs
    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()