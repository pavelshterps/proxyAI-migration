from pydantic import BaseSettings, AnyHttpUrl
from typing import List

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 4
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 15
    CELERY_TIMEZONE: str = "UTC"

    # Uploads
    UPLOAD_FOLDER: str = "/tmp/uploads"
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1073741824
    TUS_ENDPOINT: AnyHttpUrl
    SNIPPET_FORMAT: str = "wav"

    # Models & tokens
    WHISPER_MODEL: str
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str = ""

    # Database
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
