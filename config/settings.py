from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    API_WORKERS: int = Field(1, env="API_WORKERS")
    CPU_CONCURRENCY: int = Field(1, env="CPU_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")

    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field("/tmp/results", env="RESULTS_FOLDER")
    DIARIZER_CACHE_DIR: str = Field("/tmp/diarizer_cache", env="DIARIZER_CACHE_DIR")

    WHISPER_MODEL_PATH: str = Field(
        "/hf_cache/models--guillaumekln--faster-whisper-medium",
        env="WHISPER_MODEL_PATH",
    )
    WHISPER_DEVICE: str = Field("cuda", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field("int8", env="WHISPER_COMPUTE_TYPE")
    SEGMENT_LENGTH_S: int = Field(30, env="SEGMENT_LENGTH_S")

    CELERY_BROKER_URL: str = Field("redis://redis:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(
        "redis://redis:6379/0", env="CELERY_RESULT_BACKEND"
    )
    TIMEZONE: str = Field("UTC", env="TIMEZONE")

    FLOWER_USER: str = Field("", env="FLOWER_USER")
    FLOWER_PASS: str = Field("", env="FLOWER_PASS")
    FLOWER_PORT: int = Field(5555, env="FLOWER_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()