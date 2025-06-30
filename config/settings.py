from typing import List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Tell Pydantic where to find your .env file
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # FastAPI
    FASTAPI_HOST: str = Field("0.0.0.0")
    FASTAPI_PORT: int = Field(8000)
    API_WORKERS: int = Field(1)
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # Celery / Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = Field(4)
    GPU_CONCURRENCY: int = Field(1)
    CELERY_RESULT_EXPIRES: int = Field(3600)

    # Storage
    UPLOAD_FOLDER: str
    RESULTS_FOLDER: str
    FILE_RETENTION_DAYS: int = Field(7)
    MAX_FILE_SIZE: int = Field(1_073_741_824)

    # tusd
    TUSD_ENDPOINT: str
    SNIPPET_FORMAT: str = Field("wav")

    # pyannote
    DIARIZER_CACHE_DIR: str
    PYANNOTE_PROTOCOL: str

    # Hugging Face
    HUGGINGFACE_TOKEN: str
    HF_CACHE_DIR: str

    # Whisper / faster-whisper
    WHISPER_MODEL_PATH: str
    WHISPER_DEVICE: str = Field("cuda")
    WHISPER_DEVICE_INDEX: int = Field(0)
    WHISPER_COMPUTE_TYPE: str = Field("int8")
    WHISPER_BEAM_SIZE: int = Field(5)
    WHISPER_TASK: str = Field("transcribe")

    # Cleanup
    CLEAN_UP_UPLOADS: bool = Field(True)

    # Database / misc
    DATABASE_URL: str
    REDIS_URL: str


settings = Settings()