# config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    API_WORKERS: int = 1
    CPU_CONCURRENCY: int = 1
    GPU_CONCURRENCY: int = 1
    WHISPER_MODEL_PATH: str = "/hf_cache/models--guillaumekln--faster-whisper-medium"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "int8"
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"
    SEGMENT_LENGTH_S: int = 30
    TIMEZONE: str = "UTC"
    FLOWER_USER: str
    FLOWER_PASS: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()