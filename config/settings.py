from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # upload & results
    UPLOAD_FOLDER: str = Field(..., env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field(..., env="RESULTS_FOLDER")
    FILE_RETENTION_DAYS: int = Field(7, env="FILE_RETENTION_DAYS")
    MAX_FILE_SIZE: int = Field(1073741824, env="MAX_FILE_SIZE")

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CPU_CONCURRENCY: int = Field(4, env="CPU_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")

    # Whisper
    WHISPER_DEVICE: str = Field("cpu", env="WHISPER_DEVICE")
    WHISPER_DEVICE_INDEX: int = Field(0, env="WHISPER_DEVICE_INDEX")
    WHISPER_COMPUTE_TYPE: str = Field("float32", env="WHISPER_COMPUTE_TYPE")
    WHISPER_INTER_THREADS: int = Field(1, env="WHISPER_INTER_THREADS")
    WHISPER_INTRA_THREADS: int = Field(1, env="WHISPER_INTRA_THREADS")
    WHISPER_BEAM_SIZE: int = Field(5, env="WHISPER_BEAM_SIZE")
    WHISPER_BEST_OF: int = Field(5, env="WHISPER_BEST_OF")
    WHISPER_TASK: str = Field("transcribe", env="WHISPER_TASK")
    WHISPER_MODEL_PATH: str = Field(..., env="WHISPER_MODEL_PATH")

    # Pyannote diarizer
    DIARIZER_CACHE_DIR: str = Field(..., env="DIARIZER_CACHE_DIR")
    PYANNOTE_PROTOCOL: str = Field(..., env="PYANNOTE_PROTOCOL")

    # Hugging Face
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")
    HF_CACHE_DIR: str = Field(..., env="HF_CACHE_DIR")

    # Tusd
    TUSD_ENDPOINT: str = Field(..., env="TUSD_ENDPOINT")
    SNIPPET_FORMAT: str = Field(..., env="SNIPPET_FORMAT")

    # DB
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()