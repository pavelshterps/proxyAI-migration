import json
from typing import List, Optional, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # версия приложения
    APP_VERSION: str = Field("0.0.0", env="APP_VERSION")
    ADMIN_API_KEY: str = Field(..., env="ADMIN_API_KEY")

    # база данных
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    POSTGRES_DB: str = Field(..., env="POSTGRES_DB")
    POSTGRES_USER: str = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")

    # Celery / Redis
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")
    CELERY_TIMEZONE: str = Field('UTC', env="CELERY_TIMEZONE")

    CELERY_SENTINELS: List[Tuple[str, int]] = Field(
        default=[('sentinel1', 26379), ('sentinel2', 26379)],
        env='CELERY_SENTINELS',
    )
    CELERY_SENTINEL_MASTER_NAME: str = Field('mymaster', env='CELERY_SENTINEL_MASTER_NAME')
    CELERY_SENTINEL_SOCKET_TIMEOUT: float = Field(0.1, env='CELERY_SENTINEL_SOCKET_TIMEOUT')

    # concurrency
    API_WORKERS: int = Field(1, env="API_WORKERS")
    CPU_CONCURRENCY: int = Field(1, env="CPU_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")
    FFMPEG_THREADS: int = 4

    # пути
    UPLOAD_FOLDER: str = Field(..., env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field(..., env="RESULTS_FOLDER")
    DIARIZER_CACHE_DIR: str = Field(..., env="DIARIZER_CACHE_DIR")

    # Whisper
    WHISPER_MODEL_PATH: str = Field(..., env="WHISPER_MODEL_PATH")
    WHISPER_DEVICE: str = Field(..., env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(..., env="WHISPER_COMPUTE_TYPE")
    WHISPER_BATCH_SIZE: int = Field(1, env="WHISPER_BATCH_SIZE")
    WHISPER_LANGUAGE: Optional[str] = Field(None, env="WHISPER_LANGUAGE")
    PREVIEW_WHISPER_MODEL: Optional[str] = Field(None, env="PREVIEW_WHISPER_MODEL")

    # HF cache & token
    HUGGINGFACE_CACHE_DIR: Optional[str] = Field(None, env="HUGGINGFACE_CACHE_DIR")
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")

    # Pyannote / VAD / Diarization
    VAD_MODEL_PATH: str = Field(..., env="VAD_MODEL_PATH")
    PYANNOTE_PIPELINE: str = Field(..., env="PYANNOTE_PIPELINE")
    PREVIEW_LENGTH_S: int = Field(60, env="PREVIEW_LENGTH_S")
    CHUNK_LENGTH_S: int = Field(300, env="CHUNK_LENGTH_S")

    # FS-EEND (опционально)
    USE_FS_EEND: bool = Field(False, env="USE_FS_EEND")
    FS_EEND_MODEL_PATH: Optional[str] = Field(None, env="FS_EEND_MODEL_PATH")
    FS_EEND_DEVICE: str = Field("cuda", env="FS_EEND_DEVICE")
    FRAME_SHIFT: float = Field(0.01, env="FRAME_SHIFT")

    # другие настройки
    MAX_FILE_SIZE: int = Field(1_073_741_824, env="MAX_FILE_SIZE")
    FILE_RETENTION_DAYS: int = Field(7, env="FILE_RETENTION_DAYS")
    TUS_ENDPOINT: str = Field(..., env="TUS_ENDPOINT")
    ALLOWED_ORIGINS: str = Field('["*"]', env="ALLOWED_ORIGINS")
    FLOWER_USER: Optional[str] = Field(None, env="FLOWER_USER")
    FLOWER_PASS: Optional[str] = Field(None, env="FLOWER_PASS")
    EXTERNAL_API_URL: str = Field(..., env="EXTERNAL_API_URL")
    EXTERNAL_API_KEY: str = Field(..., env="EXTERNAL_API_KEY")
    EXTERNAL_POLL_INTERVAL_S: int = Field(5, env="EXTERNAL_POLL_INTERVAL_S")
    DEFAULT_TRANSCRIBE_MODE: str = Field("local", env="DEFAULT_TRANSCRIBE_MODE")

    @property
    def ALLOWED_ORIGINS_LIST(self) -> List[str]:
        try:
            return json.loads(self.ALLOWED_ORIGINS)
        except Exception:
            return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

settings = Settings()