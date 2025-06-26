# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Where uploaded WAVs live
    UPLOAD_FOLDER: str

    # TUS protocol file server (tusd) endpoint, e.g. http://tusd:1080/files/
    TUSD_ENDPOINT: str

    # Celery broker and backend URLs
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # Redis URL for fast lookup
    REDIS_URL: str

    # Whisper settings
    WHISPER_MODEL: str
    WHISPER_DEVICE: str
    WHISPER_COMPUTE_TYPE: str

    # Pyannote diarization model
    PYANNOTE_MODEL: str

    # Concurrency
    CPU_CONCURRENCY: int
    GPU_CONCURRENCY: int

    PYANNOTE_PROTOCOL: str
    # Diarization chunk length (seconds)
    DIARIZE_CHUNK_LENGTH: int
    MAX_FILE_SIZE: int
    # Other settings if present…
    # …

# instantiate once
settings = Settings()