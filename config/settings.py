from pydantic_settings import BaseSettings, SettingsConfigDict, Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # FastAPI
    FASTAPI_HOST: str = Field("0.0.0.0", env="FASTAPI_HOST")
    FASTAPI_PORT: int = Field(8000, env="FASTAPI_PORT")
    API_WORKERS: int = Field(1, env="API_WORKERS")

    # CORS
    ALLOWED_ORIGINS: list[str] = Field(["*"], env="ALLOWED_ORIGINS")

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = Field(4, env="CELERY_CONCURRENCY")
    CELERY_TIMEZONE: str = Field("UTC", env="CELERY_TIMEZONE")

    # Uploads
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int = Field(7, env="FILE_RETENTION_DAYS")
    MAX_FILE_SIZE: int = Field(1073741824, env="MAX_FILE_SIZE")
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str = Field("wav", env="SNIPPET_FORMAT")

    # Models
    DEVICE: str = Field("cuda", env="DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field("float16", env="WHISPER_COMPUTE_TYPE")
    WHISPER_MODEL: str
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int = Field(5, env="ALIGN_BEAM_SIZE")
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str
    HF_TOKEN: str

    # DB
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str
    REDIS_URL: str

settings = Settings()