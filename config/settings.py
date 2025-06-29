# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST:       str = "0.0.0.0"
    FASTAPI_PORT:       int = 8000
    API_WORKERS:        int = 1
    ALLOWED_ORIGINS:    List[str] = ["*"]

    # Celery / Redis
    CELERY_BROKER_URL:      str
    CELERY_RESULT_BACKEND:  str
    CPU_CONCURRENCY:        int = 4
    GPU_CONCURRENCY:        int = 1
    TIMEZONE:               str = "UTC"

    # Storage
    UPLOAD_FOLDER:      str = "/tmp/uploads"
    RESULTS_FOLDER:     str = "/tmp/results"
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE:      int = 1 << 30  # 1 GiB

    # tusd
    TUSD_ENDPOINT:      str
    SNIPPET_FORMAT:     str = "wav"

    # pyannote diarizer
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"
    PYANNOTE_PROTOCOL:  str = "pyannote/speaker-diarization"

    # Hugging Face
    HUGGINGFACE_TOKEN:  str
    HF_CACHE_DIR:       str = "/hf_cache"

    # Whisper / faster-whisper
    WHISPER_MODEL_PATH:    str = "/hf_cache/models--guillaumekln--faster-whisper-medium"
    WHISPER_DEVICE:        str = "cuda"
    WHISPER_DEVICE_INDEX:  int = 0
    WHISPER_COMPUTE_TYPE:  str = "int8"
    WHISPER_BEAM_SIZE:     int = 5
    WHISPER_LANGUAGE:      str = "ru"
    WHISPER_TASK:          str = "transcribe"
    SEGMENT_LENGTH_S:      int = 30

    # Cleanup
    CLEAN_UP_UPLOADS:   bool = True

    model_config = SettingsConfigDict(extra="allow")

settings = Settings()