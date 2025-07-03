# config/settings.py

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # версия приложения, подхватывается из APP_VERSION
    APP_VERSION: str = Field("0.0.0", env="APP_VERSION")

    ADMIN_API_KEY: str = Field(...)
    DATABASE_URL: str = Field(..., env='DATABASE_URL')
    POSTGRES_NAME: str = Field(...)
    POSTGRES_USER: str = Field(...)
    POSTGRES_PASSWORD: str = Field(...)

    CELERY_BROKER_URL: str = Field(...)
    CELERY_RESULT_BACKEND: str = Field(...)
    CELERY_TIMEZONE: str = Field('UTC')

    API_WORKERS: int = Field(1)
    CPU_CONCURRENCY: int = Field(1)
    GPU_CONCURRENCY: int = Field(1)

    UPLOAD_FOLDER: str = Field(...)
    RESULTS_FOLDER: str = Field(...)
    DIARIZER_CACHE_DIR: str = Field(...)

    # --------- Whisper settings ------------
    WHISPER_MODEL_PATH: str = Field(..., env="WHISPER_MODEL_PATH")
    WHISPER_DEVICE: str = Field(..., env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(..., env="WHISPER_COMPUTE_TYPE")
    WHISPER_BATCH_SIZE: int = Field(1, env="WHISPER_BATCH_SIZE")
    WHISPER_LANGUAGE: Optional[str] = Field("en", env="WHISPER_LANGUAGE")

    HUGGINGFACE_CACHE_DIR: Optional[str] = Field(None, env="HUGGINGFACE_CACHE_DIR")

    # -------- Pyannote ------------
    PYANNOTE_PIPELINE: str = Field(..., env="PYANNOTE_PIPELINE")
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")

    SEGMENT_LENGTH_S: int = Field(30, env="SEGMENT_LENGTH_S")
    VAD_LEVEL: int = Field(2, env="VAD_LEVEL")

    MAX_FILE_SIZE: int = Field(1_073_741_824, env="MAX_FILE_SIZE")
    FILE_RETENTION_DAYS: int = Field(7, env="FILE_RETENTION_DAYS")

    TUS_ENDPOINT: str = Field(..., env="TUS_ENDPOINT")

    ALLOWED_ORIGINS: str = Field('["*"]', env="ALLOWED_ORIGINS")
    FLOWER_USER: Optional[str] = Field(None, env="FLOWER_USER")
    FLOWER_PASS: Optional[str] = Field(None, env="FLOWER_PASS")

    @property
    def ALLOWED_ORIGINS_LIST(self) -> List[str]:
        import json
        try:
            return json.loads(self.ALLOWED_ORIGINS)
        except Exception:
            return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

settings = Settings()