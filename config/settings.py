from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    FASTAPI_HOST: str
    FASTAPI_PORT: int
    API_WORKERS: int

    ALLOWED_ORIGINS: list[str]

    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int
    GPU_CONCURRENCY: int

    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int
    MAX_FILE_SIZE: int
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str

    DEVICE: str
    WHISPER_MODEL: str
    WHISPER_COMPUTE_TYPE: str
    ALIGN_MODEL_NAME: str
    ALIGN_BEAM_SIZE: int
    PYANNOTE_PROTOCOL: str
    HUGGINGFACE_TOKEN: str

    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str
    REDIS_URL: str

settings = Settings()