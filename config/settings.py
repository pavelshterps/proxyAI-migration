from pydantic import BaseSettings


class Settings(BaseSettings):
    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # FastAPI
    API_WORKERS: int = 1

    # Concurrency
    CPU_CONCURRENCY: int = 1
    GPU_CONCURRENCY: int = 1

    # Whisper
    WHISPER_MODEL_PATH: str = "/hf_cache/models--guillaumekln--faster-whisper-medium"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Pyannote
    DIARIZER_CACHE_DIR: str = "/tmp/diarizer_cache"

    # VAD
    VAD_AGGRESSIVENESS: int = 3
    VAD_FRAME_MS: int = 30
    VAD_PADDING_MS: int = 300

    # Audio segmentation
    SEGMENT_LENGTH_S: int = 30  # fallback

    # Timezone
    TIMEZONE: str = "UTC"

    # Flower auth
    FLOWER_USER: str
    FLOWER_PASS: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()