# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API (FastAPI/Uvicorn)
    API_WORKERS: int = Field(2, env="API_WORKERS")

    # Celery
    CELERY_CONCURRENCY: int = Field(2, env="CELERY_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")

    # Storage paths
    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field("/tmp/results", env="RESULTS_FOLDER")

    # Diarizer cache (должен быть доступен на запись)
    DIARIZER_CACHE_DIR: str = Field("/tmp/diarizer_cache", env="DIARIZER_CACHE_DIR")

    # Faster-Whisper / Whisper модель
    WHISPER_MODEL_PATH: str = Field(
        "/hf_cache/models--guillaumekln--faster-whisper-medium",
        env="WHISPER_MODEL_PATH"
    )
    WHISPER_DEVICE: str = Field("cuda", env="WHISPER_DEVICE")
    WHISPER_DEVICE_INDEX: int = Field(0, env="WHISPER_DEVICE_INDEX")
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE")

    # Optional: настройки beam search
    WHISPER_BEAM_SIZE: int = Field(5, env="WHISPER_BEAM_SIZE")

    # Pyannote diarization model
    PYANNOTE_PROTOCOL: str = Field("pyannote/speaker-diarization", env="PYANNOTE_PROTOCOL")

    # Hugging Face token (для приватных моделей)
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")

    model_config = Settings.ConfigDict(
        env_file=".env",
        extra="ignore"
    )

# сразу создаём инстанс, чтобы celery_app и main.py могли его импортировать
settings = Settings()