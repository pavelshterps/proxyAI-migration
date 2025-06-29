# config/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_path: Path
    hf_cache_dir: Path
    whisper_device: str
    whisper_compute_type: str
    whisper_beam_size: int

    pyannote_protocol: str
    diarizer_cache_dir: Path

    upload_folder: Path
    results_folder: Path
    segment_length_s: int

    celery_broker_url: str
    celery_result_backend: str
    cpu_concurrency: int
    gpu_concurrency: int
    timezone: str

    # secrets (in prod, pull from Docker Secrets or Vault)
    huggingface_token: str

    # optional feature flags
    enable_metrics: bool = True
    clean_up_uploads: bool = True
    file_retention_days: int = 7
    max_file_size: int = 1_073_741_824  # 1 GiB

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()