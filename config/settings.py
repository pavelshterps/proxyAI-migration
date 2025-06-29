from pydantic import BaseSettings

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Celery & Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1

    # File storage
    UPLOAD_FOLDER: str = "/tmp/uploads"
    RESULTS_FOLDER: str = "/tmp/results"
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1 * 1024 * 1024 * 1024  # 1 GiB

    # tusd (resumable upload)
    TUSD_ENDPOINT: str
    SNIPPET_FORMAT: str = "wav"

    # pyannote cache
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"
    PYANNOTE_PROTOCOL: str

    # Hugging Face
    HUGGINGFACE_TOKEN: str
    HF_CACHE_DIR: str = "/hf_cache"

    # Faster Whisper / Whisper
    WHISPER_MODEL_PATH: str
    WHISPER_DEVICE: str = "cuda"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_TASK: str = "transcribe"

    # Cleanup
    CLEAN_UP_UPLOADS: bool = True

    # Database / Redis (если используются)
    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()