import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import List, Optional
from urllib.parse import urlparse

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ADMIN_API_KEY: str = Field(...)
    DATABASE_URL: str = Field(..., env='DATABASE_URL')

    CELERY_BROKER_URL: str = Field(...)
    CELERY_RESULT_BACKEND: str = Field(...)
    CELERY_TIMEZONE: str = Field('UTC')

    API_WORKERS: int = Field(1)
    CPU_CONCURRENCY: int = Field(1)
    GPU_CONCURRENCY: int = Field(1)

    UPLOAD_FOLDER: str = Field(...)
    RESULTS_FOLDER: str = Field(...)
    DIARIZER_CACHE_DIR: str = Field(...)

    WHISPER_MODEL_PATH: str = Field(...)
    WHISPER_DEVICE: str = Field(...)
    WHISPER_COMPUTE_TYPE: str = Field(...)
    PYANNOTE_PROTOCOL: str = Field(...)
    HUGGINGFACE_TOKEN: str = Field(...)

    SEGMENT_LENGTH_S: int = Field(30)
    VAD_LEVEL: int = Field(2)

    MAX_FILE_SIZE: int = Field(1073741824)
    FILE_RETENTION_DAYS: int = Field(7)

    TUS_ENDPOINT: str = Field(...)

    ALLOWED_ORIGINS: str = Field('["*"]', env="ALLOWED_ORIGINS")

    FLOWER_USER: Optional[str] = Field(None)
    FLOWER_PASS: Optional[str] = Field(None)

    @property
    def ALLOWED_ORIGINS_LIST(self) -> List[str]:
        import json
        try:
            return json.loads(self.ALLOWED_ORIGINS)
        except Exception:
            return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

settings = Settings()