from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # how many Uvicorn workers for the API
    API_WORKERS: int = Field(1, env="API_WORKERS")
    # concurrency settings for Celery
    CELERY_CONCURRENCY: int = Field(1, env="CELERY_CONCURRENCY")
    GPU_CONCURRENCY: int = Field(1, env="GPU_CONCURRENCY")

    # Redis broker / backend URLs
    CELERY_BROKER_URL: str = Field("redis://redis:6379//", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field("redis://redis:6379/", env="CELERY_RESULT_BACKEND")

    # file paths
    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    RESULTS_FOLDER: str = Field("/tmp/results", env="RESULTS_FOLDER")

    # load .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

# instantiate a single settings object
settings = Settings()