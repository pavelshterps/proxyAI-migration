from pydantic import BaseSettings, HttpUrl
from typing import List

class Settings(BaseSettings):
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 4
    ALLOWED_ORIGINS: List[str] = ["*"]

    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 15
    CELERY_TIMEZONE: str = "UTC"

    UPLOAD_FOLDER: str = "/tmp/uploads"
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1073741824
    TUS_ENDPOINT: HttpUrl
    SNIPPET_FORMAT: str = "wav"

    WHISPER_MODEL: str
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str = ""

    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
