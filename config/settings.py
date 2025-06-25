from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # FastAPI
    UPLOAD_FOLDER: str
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    # Pyannote
    PYANNOTE_MODEL: str

    # Whisper
    WHISPER_MODEL: str
    WHISPER_COMPUTE_TYPE: str = "float16"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_INTER_THREADS: int = 1
    WHISPER_INTRA_THREADS: int = 1

    # Cache
    HF_CACHE: str = "/hf_cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()