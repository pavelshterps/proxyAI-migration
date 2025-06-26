from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Tell Pydantic where to pull env from
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # FastAPI
    fastapi_host: str = Field("0.0.0.0", env="FASTAPI_HOST")
    fastapi_port: int = Field(8000, env="FASTAPI_PORT")
    api_workers: int = Field(1, env="API_WORKERS")
    allowed_origins: list[str] = Field(["*"], env="ALLOWED_ORIGINS")

    # Celery / Redis
    celery_broker_url: str = Field(..., env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(..., env="CELERY_RESULT_BACKEND")
    celery_concurrency: int = Field(1, env="CELERY_CONCURRENCY")
    celery_timezone: str = Field("UTC", env="CELERY_TIMEZONE")

    # File uploads
    upload_folder: str = Field(..., env="UPLOAD_FOLDER")
    file_retention_days: int = Field(..., env="FILE_RETENTION_DAYS")
    max_file_size: int = Field(..., env="MAX_FILE_SIZE")
    tus_endpoint: str = Field(..., env="TUS_ENDPOINT")
    snippet_format: str = Field("wav", env="SNIPPET_FORMAT")

    # Whisper
    device: str = Field("cpu", env="DEVICE")
    whisper_compute_type: str = Field("float16", env="WHISPER_COMPUTE_TYPE")
    whisper_model: str = Field(..., env="WHISPER_MODEL")
    align_model_name: str = Field(..., env="ALIGN_MODEL_NAME")
    align_beam_size: int = Field(5, env="ALIGN_BEAM_SIZE")

    # Pyannote
    pyannote_protocol: str = Field(..., env="PYANNOTE_PROTOCOL")
    pyannote_model: str | None = Field(None, env="PYANNOTE_MODEL")

    # HF tokens
    huggingface_token: str = Field(..., env="HUGGINGFACE_TOKEN")
    hf_token: str | None = Field(None, env="HF_TOKEN")

    # Postgres / SQLAlchemy
    postgres_db: str = Field(..., env="POSTGRES_DB")
    postgres_user: str = Field(..., env="POSTGRES_USER")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")
    database_url: str = Field(..., env="DATABASE_URL")

    # Redis (if used in code)
    redis_url: str = Field(..., env="REDIS_URL")


settings = Settings()