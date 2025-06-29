# config/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Разрешаем незадекларированные поля игнорировать,
    # чтобы новые переменные из .env не падали валидатором
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Celery / Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1
    TIMEZONE: str = "UTC"

    # File storage
    UPLOAD_FOLDER: Path = Path("/tmp/uploads")
    RESULTS_FOLDER: Path = Path("/tmp/results")
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1 * 1024**3  # 1 GiB

    # tusd (resumable upload)
    TUSD_ENDPOINT: str
    SNIPPET_FORMAT: str = "wav"

    # Pyannote diarizer cache
    DIARIZER_CACHE_DIR: Path = Path("/tmp/diarizer_cache")
    PYANNOTE_PROTOCOL: str

    # Hugging Face
    HUGGINGFACE_TOKEN: str
    HF_CACHE_DIR: Path = Path("/hf_cache")

    # Whisper model (faster-whisper)
    WHISPER_MODEL_PATH: Path
    WHISPER_DEVICE: str = "cuda"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_TASK: str = "transcribe"
    # ← добавляем сюда env-переменную, которую таски по-тему ожидают:
    SEGMENT_LENGTH_S: int = 30

    # Cleanup
    CLEAN_UP_UPLOADS: bool = True

    # Database
    DATABASE_URL: str
    REDIS_URL: str

    # Настройки CTranslate2 (если нужны)
    # CTR_TRANSLATE2_CACHE: Path = Path.home() / ".cache" / "ctranslate2"


# Синглтон-конфиг, импортируем из кода как `settings`
settings = Settings()