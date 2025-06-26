# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    CELERY_BROKER_URL: str = Field("redis://redis:6379", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field("redis://redis:6379", env="CELERY_RESULT_BACKEND")
    PYANNOTE_MODEL: str = Field("pyannote/speaker-diarization", env="PYANNOTE_MODEL")
    WHISPER_MODEL: str = Field("openai/whisper-large-v2", env="WHISPER_MODEL")

    # ресурсы воркеров
    CPU_CONCURRENCY: int = Field(4, env="WORKER_CPU_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="WORKER_GPU_CONCURRENCY")

    # остальные переменные (HTTP, порты и пр.) — только если вы ими реально пользуетесь в коде
    FASTAPI_HOST: str = Field("0.0.0.0", env="FASTAPI_HOST")
    FASTAPI_PORT: int = Field(8000, env="FASTAPI_PORT")

settings = Settings()