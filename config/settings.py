from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: str = '["*"]'

    # Celery & Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1

    # File storage
    UPLOAD_FOLDER: str
    RESULTS_FOLDER: str

    # Retention
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1073741824

    # tusd
    TUSD_ENDPOINT: str
    SNIPPET_FORMAT: str = "wav"

    # caches
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"
    HF_CACHE_DIR: str

    # HuggingFace
    HUGGINGFACE_TOKEN: str

    # Whisper
    WHISPER_MODEL_PATH: str
    WHISPER_DEVICE: str = "cuda"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 5

    # Pyannote
    PYANNOTE_PROTOCOL: str

    CLEAN_UP_UPLOADS: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()