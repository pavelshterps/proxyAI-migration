import os
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Upload / result paths
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
    RESULTS_FOLDER: str = os.getenv("RESULTS_FOLDER", "/tmp/results")

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # Concurrency
    CPU_CONCURRENCY: int = int(os.getenv("CPU_CONCURRENCY", "1"))
    GPU_CONCURRENCY: int = int(os.getenv("GPU_CONCURRENCY", "1"))

    # Model cache
    DIARIZER_CACHE_DIR: str = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
    WHISPER_MODEL_PATH: str = os.getenv("WHISPER_MODEL_PATH", "/hf_cache/models--guillaumekln--faster-whisper-medium")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    SEGMENT_LENGTH_S: int = int(os.getenv("SEGMENT_LENGTH_S", "30"))

    # App settings
    TIMEZONE: str = os.getenv("TIMEZONE", "UTC")
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()