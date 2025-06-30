# config/settings.py

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
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
    upload_folder: Path = Path("/tmp/uploads")
    results_folder: Path = Path("/tmp/results")

    # tusd
    tusd_endpoint: str
    snippet_format: str = "wav"

    # Diarizer
    diarizer_cache_dir: Path = Path("/tmp/diarizer_cache")
    pyannote_protocol: str = "pyannote/speaker-diarization"

    # Hugging Face
    huggingface_token: str
    hf_cache_dir: Path = Path("/hf_cache")

    # Whisper / Faster-Whisper
    whisper_model_path: Path
    whisper_device: str = "cuda"
    whisper_device_index: int = 0
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_language: str = "ru"
    segment_length_s: int = 30

    # Cleanup
    clean_up_uploads: bool = True
    model_task: str = "transcribe"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra_forbid=True,
    )

settings = Settings()