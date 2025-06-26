# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Куда сохраняем загруженные WAV
    UPLOAD_FOLDER: str = "/tmp/uploads"
    # Redis для Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"
    # Кэш HF-моделей (pyannote, whisper)
    HF_CACHE: str = "/hf_cache"
    # Модель для диаризации
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization"
    # Whisper-модель и параметры
    WHISPER_MODEL: str = "Systran/faster-whisper-large-v3"
    WHISPER_COMPUTE_TYPE: str = "float16"
    CHUNK_LENGTH_S: int = 30
    # Параметры конкуренции воркеров
    WORKER_CPU_CONCURRENCY: int = 4
    WORKER_GPU_CONCURRENCY: int = 1
    # Настройки Celery
    CELERY_CONCURRENCY: int = 4
    CELERY_TIMEZONE: str = "UTC"

    model_config = SettingsConfigDict(
        env_file = ".env",
        extra    = "ignore",   # игнорировать ненужные переменные
    )

# единственный экземпляр
settings = Settings()