from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    # Celery & Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    # File uploads
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str = "wav"

    # Models and keys
    DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16"
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

    # Redis fallback
    REDIS_URL: str

    # GPU worker concurrency
    GPU_CONCURRENCY: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()