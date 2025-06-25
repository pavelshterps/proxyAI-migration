from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Путь, куда сохраняются загруженные файлы
    UPLOAD_FOLDER: str = Field(..., env="UPLOAD_FOLDER")

    # Настройки Celery
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")

    # Модель и токен для pyannote (диаризация)
    PYANNOTE_MODEL: str = Field(..., env="PYANNOTE_MODEL")
    HF_TOKEN: str = Field(..., env="HF_TOKEN")

    # Модель и настройки Whisper (транскрипция)
    WHISPER_MODEL: str = Field(..., env="WHISPER_MODEL")
    WHISPER_COMPUTE_TYPE: str = Field("float16", env="WHISPER_COMPUTE_TYPE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()