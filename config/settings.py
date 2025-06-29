from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # concurrency & broker
    API_WORKERS: int = Field(2, env="API_WORKERS")
    CELERY_CONCURRENCY: int = Field(2, env="CELERY_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")
    CELERY_BROKER_URL: str = Field("redis://redis:6379/0", env="CELERY_BROKER_URL")

    # paths
    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field("/tmp/results", env="RESULTS_FOLDER")
    DIARIZER_CACHE_DIR: str = Field("/tmp/diarizer_cache", env="DIARIZER_CACHE_DIR")

    # whisper / faster-whisper model settings
    WHISPER_MODEL_PATH: str = Field(
        "/hf_cache/models--guillaumekln--faster-whisper-medium",
        env="WHISPER_MODEL_PATH"
    )
    WHISPER_DEVICE: str = Field("cuda", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE")


# instantiate immediately so celery_app and main.py can do:
settings = Settings()