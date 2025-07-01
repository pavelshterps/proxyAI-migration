import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # auth & DB
    ADMIN_API_KEY: str = Field(..., env="ADMIN_API_KEY")
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    # Celery
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")
    CELERY_TIMEZONE: str = Field("UTC", env="CELERY_TIMEZONE")

    # Concurrency
    API_WORKERS: int = Field(1, env="API_WORKERS")
    CPU_CONCURRENCY: int = Field(1, env="CPU_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")

    # Folders & retention
    UPLOAD_FOLDER: str = Field(..., env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field(..., env="RESULTS_FOLDER")
    DIARIZER_CACHE_DIR: str = Field(..., env="DIARIZER_CACHE_DIR")
    MAX_FILE_SIZE: int = Field(1_073_741_824, env="MAX_FILE_SIZE")
    FILE_RETENTION_DAYS: int = Field(7, env="FILE_RETENTION_DAYS")

    # Whisper (faster-whisper)
    WHISPER_MODEL_PATH: str = Field(..., env="WHISPER_MODEL_PATH")
    WHISPER_DEVICE: str = Field(..., env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(..., env="WHISPER_COMPUTE_TYPE")
    WHISPER_BATCH_SIZE: int = Field(1, env="WHISPER_BATCH_SIZE")
    WHISPER_LANGUAGE: str = Field("en", env="WHISPER_LANGUAGE")
    HUGGINGFACE_CACHE_DIR: Optional[str] = Field(None, env="HUGGINGFACE_CACHE_DIR")
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")

    # Pyannote
    PYANNOTE_PIPELINE: str = Field(..., env="PYANNOTE_PIPELINE")

    # Segmentation / VAD
    SEGMENT_LENGTH_S: int = Field(30, env="SEGMENT_LENGTH_S")
    VAD_LEVEL: int = Field(2, env="VAD_LEVEL")

    # TUS
    TUS_ENDPOINT: str = Field(..., env="TUS_ENDPOINT")

    # CORS
    ALLOWED_ORIGINS: str = Field('["*"]', env="ALLOWED_ORIGINS")
    FLOWER_USER: Optional[str] = Field(None, env="FLOWER_USER")
    FLOWER_PASS: Optional[str] = Field(None, env="FLOWER_PASS")

    @property
    def ALLOWED_ORIGINS_LIST(self) -> List[str]:
        try:
            return json.loads(self.ALLOWED_ORIGINS)
        except:
            return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

settings = Settings()