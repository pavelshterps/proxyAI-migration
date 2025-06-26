from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # FastAPI
    UPLOAD_FOLDER: str
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    # Pyannote
    PYANNOTE_MODEL: str  # e.g. "pyannote/speaker-diarization"

    # Whisper
    WHISPER_MODEL: str  # e.g. "openai/whisper-large-v2"
    WHISPER_COMPUTE_TYPE: str = "float16"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_INTER_THREADS: int = 1
    WHISPER_INTRA_THREADS: int = 1

    # Segmentation
    ALIGN_BEAM_SIZE: int = 5

    # Cache
    HF_CACHE: str = "/hf_cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()