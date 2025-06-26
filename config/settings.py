from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Pydantic-v2 style settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Where uploads live inside the container
    UPLOAD_FOLDER: str = "/data"

    # Celery broker & backend
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # TUS upload callback URL
    TUSD_ENDPOINT: str

    # Whisper & Pyannote settings
    WHISPER_MODEL: str = "openai/whisper-large-v2"
    WHISPER_COMPUTE_TYPE: str = "float16"
    WHISPER_BEAM_SIZE: int = 5
    PYANNOTE_PROTOCOL: str = "pyannote/speaker-diarization"
    HUGGINGFACE_TOKEN: str | None = None

    # Device and chunking
    DEVICE: str = "cuda"
    CHUNK_LENGTH_S: int = 30

    # Concurrency controls
    API_WORKERS: int = 1
    CELERY_CONCURRENCY: int = 1
    GPU_CONCURRENCY: int = 1

settings = Settings()