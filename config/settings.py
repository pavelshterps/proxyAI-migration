from functools import lru_cache
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

    # Models (quantized faster-whisper-medium by default)
    whisper_model_path: str = Field(
        "/hf_cache/models--guillaumekln--faster-whisper-medium",
        env="WHISPER_MODEL_PATH"
    )
    whisper_device: str = Field("cuda", env="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", env="WHISPER_COMPUTE_TYPE")
    pyannote_protocol: str = Field(..., env="PYANNOTE_PROTOCOL")
    huggingface_token: str = Field(..., env="HUGGINGFACE_TOKEN")

    # Segmentation / VAD
    segment_length_s: int = Field(30, gt=0, env="SEGMENT_LENGTH_S")
    vad_level: int = Field(2, ge=0, le=3, env="VAD_LEVEL")

    # File limits & retention
    max_file_size: int = Field(1_073_741_824, gt=0, env="MAX_FILE_SIZE")
    file_retention_days: int = Field(7, gt=0, env="FILE_RETENTION_DAYS")

    # Tus endpoint
    tus_endpoint: str = Field(..., env="TUS_ENDPOINT")

    # Frontend / CORS
    allowed_origins: list[str] = Field(["*"], env="ALLOWED_ORIGINS")

    # Metrics exporter
    metrics_port: int = Field(8001, gt=0, env="METRICS_PORT")

    admin_api_key: str = Field(..., env="ADMIN_API_KEY")

    # Database (SQLite for dev; override to PostgreSQL in prod)
    database_url: str = Field("sqlite+aiosqlite:///./app.db", env="DATABASE_URL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()