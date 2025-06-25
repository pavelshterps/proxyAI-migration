# config/settings.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # где-то здесь уже есть ваши поля, например:
    UPLOAD_FOLDER: str = "/tmp/uploads"
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # ----------------------------------------
    # Добавляем недостающие для pyannote.audio:
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization"
    HF_TOKEN: str
    # ----------------------------------------

    # и ваши остальные поля:
    WHISPER_MODEL: str
    WHISPER_COMPUTE_TYPE: str = "float16"
    API_WORKERS: int = 1
    # ...

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# экземпляр настроек
settings = Settings()