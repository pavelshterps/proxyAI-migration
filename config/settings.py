from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # FastAPI / Uvicorn
    API_WORKERS: int = 2

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1

    # file paths
    UPLOAD_FOLDER: str = "/tmp/uploads"
    RESULTS_FOLDER: str = "/tmp/results"

    # whisper / diarization
    WHISPER_MODEL_PATH: str = "/hf_cache/models--guillaumekln--faster-whisper-medium"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "int8"
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"
    SEGMENT_LENGTH_S: int = 30

    # HuggingFace
    HUGGINGFACE_TOKEN: str  # must be provided

    # Timezone
    TIMEZONE: str = "UTC"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()