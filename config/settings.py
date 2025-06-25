from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = Field("0.0.0.0", env="FASTAPI_HOST")
    FASTAPI_PORT: int = Field(8000, env="FASTAPI_PORT")
    API_WORKERS: int = Field(1, env="API_WORKERS")

    # CORS
    ALLOWED_ORIGINS: List[str] = Field(["*"], env="ALLOWED_ORIGINS")

    # Celery
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")
    CELERY_CONCURRENCY: int = Field(1, env="CELERY_CONCURRENCY")
    CELERY_TIMEZONE: str = Field("UTC", env="CELERY_TIMEZONE")

    # Uploads
    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    FILE_RETENTION_DAYS: int = Field(7, env="FILE_RETENTION_DAYS")
    MAX_FILE_SIZE: int = Field(1_073_741_824, env="MAX_FILE_SIZE")
    TUS_ENDPOINT: str = Field(..., env="TUS_ENDPOINT")
    SNIPPET_FORMAT: str = Field("wav", env="SNIPPET_FORMAT")

    # Whisper & diarization
    DEVICE: str = Field("cpu", env="DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field("float16", env="WHISPER_COMPUTE_TYPE")
    WHISPER_MODEL: str = Field(..., env="WHISPER_MODEL")
    ALIGN_MODEL_NAME: str = Field(..., env="ALIGN_MODEL_NAME")
    ALIGN_BEAM_SIZE: int = Field(5, env="ALIGN_BEAM_SIZE")
    PYANNOTE_PROTOCOL: str = Field(..., env="PYANNOTE_PROTOCOL")
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")

    # Database
    POSTGRES_DB: str = Field(..., env="POSTGRES_DB")
    POSTGRES_USER: str = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    # Redis
    REDIS_URL: str = Field(..., env="REDIS_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


# создаём синглтон
settings = Settings()