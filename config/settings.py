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

    ALLOWED_ORIGINS: List[str] = Field(['*'])

    FLOWER_USER: Optional[str] = Field(None)
    FLOWER_PASS: Optional[str] = Field(None)

    @validator('ALLOWED_ORIGINS', pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            # JSON format or comma-separated
            v = v.strip()
            if v.startswith('['):
                try:
                    import json
                    return json.loads(v)
                except:
                    pass
            return [i.strip() for i in v.split(',') if i.strip()]
        return v

    @validator('ALLOWED_ORIGINS', each_item=True)
    def validate_origin(cls, v):
        if v == '*':
            return v
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid origin URL: {v}")
        return v

settings = Settings()