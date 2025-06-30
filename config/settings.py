# config/settings.py

from pathlib import Path
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Redis / Celery
    celery_broker_url: str = Field(..., env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(..., env="CELERY_RESULT_BACKEND")

    # FastAPI / Uvicorn
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(1, env="API_WORKERS")

    # Worker concurrency
    cpu_concurrency: int = Field(1, env="CPU_CONCURRENCY")
    gpu_concurrency: int = Field(1, env="GPU_CONCURRENCY")

    # File paths (mounts from docker‐compose)
    upload_folder: Path = Field(Path("/tmp/uploads"), env="UPLOAD_FOLDER")
    results_folder: Path = Field(Path("/tmp/results"), env="RESULTS_FOLDER")
    diarizer_cache_dir: Path = Field(Path("/tmp/diarizer_cache"), env="DIARIZER_CACHE_DIR")
    hf_cache_dir: Path = Field(Path("/hf_cache"), env="HF_CACHE_DIR")

    # Whisper settings
    whisper_model_path: str = Field("models--guillaumekln--faster-whisper-medium", env="WHISPER_MODEL_PATH")
    whisper_device: str = Field("cuda", env="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", env="WHISPER_COMPUTE_TYPE")
    segment_length_s: int = Field(30, env="SEGMENT_LENGTH_S")

    # Timezone
    timezone: str = Field("UTC", env="TIMEZONE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# instantiate once for import
settings = Settings()