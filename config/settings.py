# config/settings.py

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = Field("0.0.0.0", description="FastAPI listen host")
    FASTAPI_PORT: int = Field(8000, description="FastAPI listen port")
    ALLOWED_ORIGINS: List[str] = Field(
        ["*"], description="CORS allowed origins"
    )

    # Celery / Redis
    CELERY_BROKER_URL: str = Field(..., description="Redis broker URL for Celery")
    CELERY_RESULT_BACKEND: str = Field(..., description="Redis backend URL for Celery results")
    CPU_CONCURRENCY: int = Field(4, description="Number of threads for cpu-worker")
    GPU_CONCURRENCY: int = Field(1, description="Number of processes for gpu-worker")
    TIMEZONE: str = Field("UTC", description="Timezone for Celery")

    # Storage
    UPLOAD_FOLDER: str = Field("/tmp/uploads", description="Host folder for incoming files")
    RESULTS_FOLDER: str = Field("/tmp/results", description="Host folder for outputs")
    FILE_RETENTION_DAYS: int = Field(7, description="Days to keep files")
    MAX_FILE_SIZE: int = Field(1_073_741_824, description="Max upload size in bytes")

    # Tusd (resumable upload)
    TUSD_ENDPOINT: str = Field(..., description="tusd files endpoint")
    SNIPPET_FORMAT: str = Field("wav", description="Format for snippet downloads")

    # Pyannote diarizer
    DIARIZER_CACHE_DIR: str = Field(
        "/tmp/diarizer_cache", description="Local cache for pyannote pipelines"
    )
    PYANNOTE_PROTOCOL: str = Field(
        "pyannote/speaker-diarization", description="Hugging Face proto for diarization"
    )

    # Hugging Face
    HUGGINGFACE_TOKEN: str = Field(..., description="HF access token (use secrets)")
    HF_CACHE_DIR: str = Field(
        "/hf_cache", description="Bind-mounted HF cache on host"
    )

    # Whisper / Faster-Whisper
    WHISPER_MODEL_PATH: str = Field(
        "/hf_cache/models--guillaumekln--faster-whisper-medium",
        description="Local path to quantized Whisper model"
    )
    WHISPER_DEVICE: str = Field("cuda", description="Device for whisper")
    WHISPER_DEVICE_INDEX: int = Field(0, description="CUDA device index")
    WHISPER_COMPUTE_TYPE: str = Field("int8", description="Quantization type")
    WHISPER_BEAM_SIZE: int = Field(5, description="Beam size for transcription")
    WHISPER_TASK: str = Field("transcribe", description="faster-whisper task")
    SEGMENT_LENGTH_S: int = Field(30, description="Fixed window length (sec)")

    # Cleanup
    CLEAN_UP_UPLOADS: bool = Field(True, description="Remove uploads after processing")

    # Database
    DATABASE_URL: str = Field(..., description="Postgres connection URL")
    REDIS_URL: str = Field(..., description="Redis URL (if used outside Celery)")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # drop any unknown vars rather than erroring


# Синглтон-настройки
settings = Settings()