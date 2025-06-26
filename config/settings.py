# config/settings.py

from typing import List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",   # игнорируем все не описанные тут переменные
    )

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Celery & Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    WORKER_CPU_CONCURRENCY: int = 4
    WORKER_GPU_CONCURRENCY: int = 1

    # Заливка файлов
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1073741824

    # TUSd
    TUS_ENDPOINT: str = "http://tusd:1080/files/"
    SNIPPET_FORMAT: str = "wav"

    # Whisper
    WHISPER_MODEL: str = "Systran/faster-whisper-large-v2"
    WHISPER_COMPUTE_TYPE: str = "float16"
    DEVICE: str = "cuda"

    # Pyannote
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization"

    # Align (опционально, если используется)
    ALIGN_MODEL_NAME: str = "whisper-large"
    ALIGN_BEAM_SIZE: int = 5

    # Huggingface
    HUGGINGFACE_TOKEN: str

    # Postgres / SQLAlchemy
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str

    # Redis (если где-то используется REDIS_URL)
    REDIS_URL: str = ""

    # Роутинг задач
    CELERY_TASK_ROUTES: Dict[str, Dict[str, str]] = {
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    }

    class Config:
        env_prefix = ""  # имя переменных точно как в .env


settings = Settings()