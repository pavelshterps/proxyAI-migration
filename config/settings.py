import os
from typing import List, Optional
from functools import lru_cache
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Admin
    ADMIN_API_KEY: str = Field(..., env='ADMIN_API_KEY')

    # Celery
    CELERY_BROKER_URL: str = Field(..., env='CELERY_BROKER_URL')
    CELERY_RESULT_BACKEND: str = Field(..., env='CELERY_RESULT_BACKEND')
    CELERY_TIMEZONE: str = Field('UTC', env='CELERY_TIMEZONE')

    # Concurrency
    API_WORKERS: int = Field(1, env='API_WORKERS')
    CPU_CONCURRENCY: int = Field(1, env='CPU_CONCURRENCY')
    GPU_CONCURRENCY: int = Field(1, env='GPU_CONCURRENCY')

    # Paths
    UPLOAD_FOLDER: str = Field(..., env='UPLOAD_FOLDER')
    RESULTS_FOLDER: str = Field(..., env='RESULTS_FOLDER')
    DIARIZER_CACHE_DIR: str = Field(..., env='DIARIZER_CACHE_DIR')

    # Models
    WHISPER_MODEL_PATH: str = Field(..., env='WHISPER_MODEL_PATH')
    WHISPER_DEVICE: str = Field(..., env='WHISPER_DEVICE')
    WHISPER_COMPUTE_TYPE: str = Field(..., env='WHISPER_COMPUTE_TYPE')
    PYANNOTE_PROTOCOL: str = Field(..., env='PYANNOTE_PROTOCOL')
    HUGGINGFACE_TOKEN: str = Field(..., env='HUGGINGFACE_TOKEN')

    # Segmentation / VAD
    SEGMENT_LENGTH_S: int = Field(30, env='SEGMENT_LENGTH_S')
    VAD_LEVEL: int = Field(2, env='VAD_LEVEL')

    # File limits & retention
    MAX_FILE_SIZE: int = Field(1073741824, env='MAX_FILE_SIZE')
    FILE_RETENTION_DAYS: int = Field(7, env='FILE_RETENTION_DAYS')

    # Tus endpoint
    TUS_ENDPOINT: str = Field(..., env='TUS_ENDPOINT')

    # Frontend/CORS
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], env='ALLOWED_ORIGINS')

    # Flower UI auth
    FLOWER_USER: Optional[str] = Field(None, env='FLOWER_USER')
    FLOWER_PASS: Optional[str] = Field(None, env='FLOWER_PASS')

    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.strip("[]").split(",") if origin.strip()]
        if isinstance(v, list):
            return v
        return []

    @validator("ALLOWED_ORIGINS", each_item=True)
    def validate_origin_url(cls, v):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid origin URL: {v}")
        return v


settings = Settings()