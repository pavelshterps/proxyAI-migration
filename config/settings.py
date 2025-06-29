# config/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Celery / Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int
    GPU_CONCURRENCY: int
    TIMEZONE: str = "UTC"

    # Storage
    UPLOAD_FOLDER: Path
    RESULTS_FOLDER: Path
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1_073_741_824

    # tusd
    TUSD_ENDPOINT: str
    SNIPPET_FORMAT: str = "wav"

    # Pyannote diarizer
    DIARIZER_CACHE_DIR: Path
    PYANNOTE_PROTOCOL: str = "pyannote/speaker-diarization"

    # Hugging Face
    HUGGINGFACE_TOKEN: str
    HF_CACHE_DIR: Path

    # Whisper / Faster-Whisper
    WHISPER_MODEL_PATH: Path
    WHISPER_DEVICE: str = "cuda"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_TASK: str = "transcribe"
    SEGMENT_LENGTH_S: int = 30

    # Cleanup
    CLEAN_UP_UPLOADS: bool = True

    # Database (if used later)
    DATABASE_URL: str | None = None
    REDIS_URL: str | None = None

    model_config = SettingsConfigDict(
        env_file = ".env",
        case_sensitive = True,
        extra = "ignore",           # allow unspecified vars without error
    )

settings = Settings()