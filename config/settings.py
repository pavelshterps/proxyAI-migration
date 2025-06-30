from typing import List
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Celery
    celery_broker_url: str = Field(..., env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(..., env="CELERY_RESULT_BACKEND")

    # API / workers
    api_workers: int = Field(1, env="API_WORKERS")
    cpu_concurrency: int = Field(1, env="CPU_CONCURRENCY")
    gpu_concurrency: int = Field(1, env="GPU_CONCURRENCY")

    # Paths
    upload_folder: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    results_folder: str = Field("/tmp/results", env="RESULTS_FOLDER")
    diarizer_cache_dir: str = Field("/tmp/diarizer_cache", env="DIARIZER_CACHE_DIR")

    # Models
    whisper_model_path: str = Field(
        "/hf_cache/models--guillaumekln--faster-whisper-medium",
        env="WHISPER_MODEL_PATH"
    )
    whisper_device: str = Field("cuda", env="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", env="WHISPER_COMPUTE_TYPE")

    # Segmentation / VAD
    segment_length_s: int = Field(30, env="SEGMENT_LENGTH_S")
    vad_level: int = Field(2, env="VAD_LEVEL")

    # File limits & retention
    max_file_size: int = Field(1073741824, env="MAX_FILE_SIZE")  # 1 GB
    file_retention_days: int = Field(7, env="FILE_RETENTION_DAYS")

    # CORS
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")

    # Tus / snippet
    tus_endpoint: str = Field(..., env="TUS_ENDPOINT")
    snippet_format: str = Field("wav", env="SNIPPET_FORMAT")

    # Metrics
    metrics_port: int = Field(8000, env="METRICS_PORT")

    # Flower auth
    flower_user: str = Field("", env="FLOWER_USER")
    flower_pass: str = Field("", env="FLOWER_PASS")

    # Timezone
    timezone: str = Field("UTC", env="CELERY_TIMEZONE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()