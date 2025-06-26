# config/settings.py
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
    )

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

    # Uploads
    UPLOAD_FOLDER: Path
    FILE_RETENTION_DAYS: int
    MAX_FILE_SIZE: int

    # TUSD
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

    # Database
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str

    # Redis (if you reference REDIS_URL)
    REDIS_URL: str

    # GPU worker
    GPU_CONCURRENCY: int

settings = Settings()