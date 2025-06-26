# config/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # 1) Подключаем .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",             # игнорируем лишние переменные
    )

    # 2) Обязательные
    UPLOAD_FOLDER: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # 3) Диаризация
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization"

    # 4) Транскрипция
    WHISPER_MODEL: str = "openai/whisper-large-v2"
    WHISPER_COMPUTE_TYPE: str = "float16"

    # 5) FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]

    # 6) Celery
    WORKER_CPU_CONCURRENCY: int = 4
    WORKER_GPU_CONCURRENCY: int = 1
    CELERY_TASK_ROUTES: dict = {
        "tasks.diarize_full": {"queue": "preprocess_cpu"},
        "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    }

    class Config:
        env_prefix = ""  # без префиксов

# Одна инстанция для всего приложения
settings = Settings()