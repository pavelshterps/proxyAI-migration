from pydantic import BaseSettings

class Settings(BaseSettings):
    # API
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: str = '["*"]'

    # Celery / Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1
    TIMEZONE: str = "UTC"

    # File storage
    UPLOAD_FOLDER: str = "/tmp/uploads"
    RESULTS_FOLDER: str = "/tmp/results"
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1_073_741_824

    # tusd
    TUSD_ENDPOINT: str

    # Pyannote
    DIARIZER_CACHE_DIR: str
    PYANNOTE_PROTOCOL: str

    # Hugging Face
    HUGGINGFACE_TOKEN: str
    HF_CACHE_DIR: str

    # Whisper
    WHISPER_MODEL_PATH: str
    WHISPER_DEVICE: str
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_COMPUTE_TYPE: str
    WHISPER_BEAM_SIZE: int
    WHISPER_TASK: str
    SEGMENT_LENGTH_S: int = 30

    # Cleanup
    CLEAN_UP_UPLOADS: bool = True

    # Database
    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()