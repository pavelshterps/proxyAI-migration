# proxyAI v13.6 – config/settings.py

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # Paths
    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field("/tmp/results", env="RESULTS_FOLDER")
    DIARIZER_CACHE_DIR: str = Field("/tmp/diarizer_cache", env="DIARIZER_CACHE_DIR")

    # Whisper
    WHISPER_MODEL_PATH: str = Field("/hf_cache/models--guillaumekln--faster-whisper-medium", env="WHISPER_MODEL_PATH")
    WHISPER_DEVICE: str = Field("cuda", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE")

    # Segmentation
    SEGMENT_LENGTH_S: int = Field(30, env="SEGMENT_LENGTH_S")

    # API & Celery
    API_WORKERS: int = Field(1, env="API_WORKERS")
    CPU_CONCURRENCY: int = Field(1, env="CPU_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")
    TIMEZONE: str = Field("UTC", env="TIMEZONE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()