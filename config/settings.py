# config/settings.py
from pydantic import BaseSettings, Field, AnyHttpUrl

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: list[str] = Field(default=["*"], env="ALLOWED_ORIGINS")

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_TIMEZONE: str = "UTC"
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1

    # Storage
    UPLOAD_FOLDER: str = "/tmp/uploads"
    RESULTS_FOLDER: str = "/tmp/results"
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1_073_741_824

    # tusd
    TUSD_ENDPOINT: AnyHttpUrl
    SNIPPET_FORMAT: str = "wav"

    # Pyannote diarizer
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"
    PYANNOTE_PROTOCOL: str

    # Hugging Face
    HUGGINGFACE_TOKEN: str
    HF_CACHE_DIR: str = "/hf_cache"

    # Whisper (faster-whisper)
    WHISPER_MODEL_PATH: str
    WHISPER_DEVICE: str = "cuda"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_TASK: str = "transcribe"
    SEGMENT_LENGTH_S: int = 30

    # Cleanup
    CLEAN_UP_UPLOADS: bool = True

    # Database
    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # lock to pydantic v1
        frozen = True

settings = Settings()