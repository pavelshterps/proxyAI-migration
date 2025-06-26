# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    ALLOWED_ORIGINS: str = '["*"]'

    # Celery / Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    # File upload
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str

    # Whisper & Pyannote
    DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    WHISPER_MODEL: str
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int = 5
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str
    HF_TOKEN: str

    # Database
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str
    REDIS_URL: str

    # GPU worker concurrency
    GPU_CONCURRENCY: int = 1

settings = Settings()