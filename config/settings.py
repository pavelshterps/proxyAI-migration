from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # CORS
    ALLOWED_ORIGINS: str = Field('["*"]', env="ALLOWED_ORIGINS")

    # Celery & Redis
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")

    API_WORKERS: int = Field(1, env="API_WORKERS")
    CPU_CONCURRENCY: int = Field(4, env="CPU_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")

    # File storage
    UPLOAD_FOLDER: str = Field(..., env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field(..., env="RESULTS_FOLDER")

    # Diarizer cache
    DIARIZER_CACHE_DIR: str = Field(..., env="DIARIZER_CACHE_DIR")

    # tusd (resumable upload)
    TUSD_ENDPOINT: str = Field(..., env="TUSD_ENDPOINT")
    SNIPPET_FORMAT: str = Field("wav", env="SNIPPET_FORMAT")

    # Hugging Face
    HUGGINGFACE_TOKEN: str | None = Field(None, env="HUGGINGFACE_TOKEN")
    HF_CACHE_DIR: str = Field(..., env="HF_CACHE_DIR")

    # Faster Whisper / Whisper
    WHISPER_MODEL_PATH: str = Field(..., env="WHISPER_MODEL_PATH")
    WHISPER_DEVICE: str = Field("cuda", env="WHISPER_DEVICE")
    WHISPER_DEVICE_INDEX: int = Field(0, env="WHISPER_DEVICE_INDEX")
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE")
    WHISPER_BEAM_SIZE: int = Field(5, env="WHISPER_BEAM_SIZE")

    # Pyannote speaker-diarization
    PYANNOTE_PROTOCOL: str = Field("pyannote/speaker-diarization", env="PYANNOTE_PROTOCOL")

    # Cleanup
    CLEAN_UP_UPLOADS: bool = Field(True, env="CLEAN_UP_UPLOADS")
    FILE_RETENTION_DAYS: int = Field(7, env="FILE_RETENTION_DAYS")
    MAX_FILE_SIZE: int = Field(1073741824, env="MAX_FILE_SIZE")

    class Config:
        env_file = ".env"

    #
    # добавляем свойства-синонимы, если кто-то в коде
    # всё ещё обращается к ним в snake_case
    #
    @property
    def celery_broker_url(self) -> str:
        return self.CELERY_BROKER_URL

    @property
    def celery_result_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND


# Единый инстанс на всё приложение
settings = Settings()