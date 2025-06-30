# config/settings.py
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Celery
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")

    # FastAPI
    API_WORKERS: int = Field(1, env="API_WORKERS")
    TIMEZONE: str = Field("UTC", env="TIMEZONE")

    # Paths
    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field("/tmp/results", env="RESULTS_FOLDER")

    # Whisper
    WHISPER_MODEL_PATH: str = Field("/hf_cache/models--guillaumekln--faster-whisper-medium",
                                    env="WHISPER_MODEL_PATH")
    WHISPER_DEVICE: str = Field("cuda", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE")

    # DIARIZER
    DIARIZER_CACHE_DIR: str = Field("/tmp/diarizer_cache", env="DIARIZER_CACHE_DIR")
    SEGMENT_LENGTH_S: int = Field(30, env="SEGMENT_LENGTH_S")

    # Flower
    FLOWER_USER: str = Field("", env="FLOWER_USER")
    FLOWER_PASS: str = Field("", env="FLOWER_PASS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()