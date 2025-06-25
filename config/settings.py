# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # path
    UPLOAD_FOLDER: str
    HF_HOME: str = "/hf_cache"

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 4

    # Pyannote diarization
    PYANNOTE_MODEL: str
    HUGGINGFACE_TOKEN: str

    # Whisper
    WHISPER_MODEL: str
    WHISPER_COMPUTE_TYPE: str = "float16"
    ALIGN_BEAM_SIZE: int = 5

    model_config = {
      "env_file": ".env",
      "extra": "ignore"
    }

settings = Settings()