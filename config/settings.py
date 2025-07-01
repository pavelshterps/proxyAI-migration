import json
from functools import lru_cache
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Celery
    celery_broker_url: str = Field(..., env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(..., env="CELERY_RESULT_BACKEND")
    timezone: str = Field("UTC", env="CELERY_TIMEZONE")

    # Concurrency
    api_workers: int = Field(1, env="API_WORKERS")
    cpu_concurrency: int = Field(1, env="CPU_CONCURRENCY")
    gpu_concurrency: int = Field(1, env="GPU_CONCURRENCY")

    # Paths
    upload_folder: str = Field("/data/uploads", env="UPLOAD_FOLDER")
    results_folder: str = Field("/data/results", env="RESULTS_FOLDER")
    diarizer_cache_dir: str = Field("/data/diarizer_cache", env="DIARIZER_CACHE_DIR")
    hf_cache_dir: str = Field("/hf_cache", env="HF_CACHE_DIR")

    # Models
    whisper_model_path: str = Field(
        "/hf_cache/models--guillaumekln--faster-whisper-medium",
        env="WHISPER_MODEL_PATH"
    )
    whisper_device: str = Field("cuda", env="WHISPER_DEVICE")
    whisper_device_index: int = Field(0, env="WHISPER_DEVICE_INDEX")
    whisper_compute_type: str = Field("int8", env="WHISPER_COMPUTE_TYPE")
    whisper_beam_size: int = Field(5, ge=1, env="WHISPER_BEAM_SIZE")

    pyannote_protocol: str = Field(..., env="PYANNOTE_PROTOCOL")
    huggingface_token: str = Field(..., env="HUGGINGFACE_TOKEN")

    # Segmentation / VAD
    segment_length_s: int = Field(30, gt=0, env="SEGMENT_LENGTH_S")
    vad_threshold: float = Field(0.35, env="VAD_THRESHOLD")
    vad_min_duration_on: float = Field(0.5, env="VAD_MIN_DURATION_ON")
    vad_min_duration_off: float = Field(0.3, env="VAD_MIN_DURATION_OFF")

    # File Handling
    tusd_endpoint: str = Field(..., env="TUSD_ENDPOINT")
    snippet_format: str = Field("wav", env="SNIPPET_FORMAT")
    clean_up_uploads: bool = Field(True, env="CLEAN_UP_UPLOADS")
    file_retention_days: int = Field(7, env="FILE_RETENTION_DAYS")
    max_file_size: int = Field(1073741824, env="MAX_FILE_SIZE")  # 1GB

    # CORS
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"], env="ALLOWED_ORIGINS")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                # fallback: comma-separated string
                return [s.strip() for s in v.split(",") if s.strip()]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()