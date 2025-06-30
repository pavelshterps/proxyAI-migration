from typing import List
from pydantic import BaseSettings, Field, validator

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

    # Models
    whisper_model_path: str = Field(..., env="WHISPER_MODEL_PATH")
    whisper_device: str = Field("cuda", env="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", env="WHISPER_COMPUTE_TYPE")
    pyannote_protocol: str = Field(..., env="PYANNOTE_PROTOCOL")
    huggingface_token: str = Field(..., env="HUGGINGFACE_TOKEN")

    # Segmentation / VAD
    segment_length_s: int = Field(30, gt=0, env="SEGMENT_LENGTH_S")
    vad_level: int = Field(2, ge=0, le=3, env="VAD_LEVEL")

    # File limits & retention
    max_file_size: int = Field(1_073_741_824, gt=0, env="MAX_FILE_SIZE")  # 1 GB
    file_retention_days: int = Field(7, gt=0, env="FILE_RETENTION_DAYS")

    # Tus
    tus_endpoint: str = Field(..., env="TUS_ENDPOINT")

    # Frontend
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("allowed_origins", pre=True)
    def split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

settings = Settings()