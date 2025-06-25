from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Подхватываем .env
load_dotenv()

class Settings(BaseSettings):
    # Игнорировать неизвестные поля
    model_config = SettingsConfigDict(extra="ignore")

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Celery & Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 1
    CELERY_TIMEZONE: str = "UTC"

    # Заливка файлов
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int

    # TUS (если используется)
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str

    # Модели и токены
    DEVICE: str
    WHISPER_COMPUTE_TYPE: str
    WHISPER_MODEL: str
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str

    # Postgres (если используется)
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str

    # Альтернативный URL для Redis
    REDIS_URL: str

    # Параметры threading для Whisper
    INTER_THREADS: int = 1
    INTRA_THREADS: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()