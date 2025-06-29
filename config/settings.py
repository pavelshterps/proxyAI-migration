# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # где лежат загруженные WAV-файлы и куда писать результаты
    UPLOAD_FOLDER: str = "/tmp/uploads"
    RESULTS_FOLDER: str = "/tmp/results"

    # tusd
    TUSD_ENDPOINT: str

    # celery + redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    REDIS_URL: str

    # кэш для Pyannote
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"

    # whisper
    WHISPER_MODEL_PATH: str
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_DEVICE_INDEX: int = 0

    # параллелизм
    CPU_CONCURRENCY: int = 1
    GPU_CONCURRENCY: int = 1

    # модель для Pyannote
    PYANNOTE_MODEL: str

settings = Settings()