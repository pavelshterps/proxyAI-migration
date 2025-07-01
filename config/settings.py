import os
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, Field, validator

class Settings(BaseSettings):
    # ProxyAI 13.7.4 example environment
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
    ALLOWED_ORIGINS: str = Field('["*"]', env='ALLOWED_ORIGINS')

    # Flower UI auth
    FLOWER_USER: Optional[str] = Field(None, env='FLOWER_USER')
    FLOWER_PASS: Optional[str] = Field(None, env='FLOWER_PASS')

    @property
    def allowed_origins_list(self) -> List[str]:
        # Преобразует строку '["*"]' или '["http://localhost"]' в список Python
        import json
        try:
            return json.loads(self.ALLOWED_ORIGINS)
        except Exception:
            # fallback: разделение по запятой (например: 'http://localhost,http://127.0.0.1')
            return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()