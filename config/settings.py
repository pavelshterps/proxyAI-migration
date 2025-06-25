from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # читаем .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    # File upload
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str = "wav"

    # Models & tokens
    DEVICE: str = "cpu"
    WHISPER_MODEL: str
    WHISPER_COMPUTE_TYPE: str = "float16"
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int = 5
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str

    # Database
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str
    REDIS_URL: str = ""

settings = Settings()