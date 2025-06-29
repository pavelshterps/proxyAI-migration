from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model: SettingsConfigDict = SettingsConfigDict(env_file=".env", extra="ignore")

    # FastAPI
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    api_workers: int = 1
    allowed_origins: List[str] = ["*"]

    # Celery / Redis
    celery_broker_url: str
    celery_result_backend: str
    cpu_concurrency: int = 4
    gpu_concurrency: int = 1
    timezone: str = "UTC"

    # Storage
    upload_folder: Path
    results_folder: Path
    file_retention_days: int = 7
    max_file_size: int = 1_073_741_824  # 1 GiB

    # tusd
    tusd_endpoint: str
    snippet_format: str = "wav"

    # Diarizer
    diarizer_cache_dir: Path
    pyannote_protocol: str = "pyannote/speaker-diarization"

    # Hugging Face
    huggingface_token: str
    hf_cache_dir: Path

    # Whisper / Faster-Whisper
    whisper_model_path: str
    whisper_device: str = "cuda"
    whisper_device_index: int = 0
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_language: str = "ru"
    segment_length_s: int = 30

    # Cleanup
    clean_up_uploads: bool = True

    # Database (future)
    database_url: str
    redis_url: str

settings = Settings()