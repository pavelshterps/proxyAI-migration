from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # FastAPI
    FASTAPI_HOST: str
    FASTAPI_PORT: int
    API_WORKERS: int

    ALLOWED_ORIGINS: list[str]

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    WORKER_CPU_CONCURRENCY: int
    WORKER_GPU_CONCURRENCY: int
    CELERY_TIMEZONE: str

    # Uploads
    UPLOAD_FOLDER: str

    # Models
    DEVICE: str
    WHISPER_COMPUTE_TYPE: str
    WHISPER_MODEL: str
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str

    # Database
    DATABASE_URL: str
    REDIS_URL: str

settings = Settings()