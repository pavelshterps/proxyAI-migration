from pydantic import BaseSettings

def parse_origins(val: str) -> list[str]:
    """
    Parse a JSON-style string from .env into a Python list.
    e.g. '["*"]' → ["*"]
    """
    import json
    try:
        return json.loads(val)
    except Exception:
        return []

class Settings(BaseSettings):
    # FastAPI
    FASTAPI_HOST: str = '0.0.0.0'
    FASTAPI_PORT: int = 8000
    API_WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: list[str] = []

    # Celery & Redis
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_CONCURRENCY: int = 4
    CELERY_TIMEZONE: str = 'UTC'

    # File uploads
    UPLOAD_FOLDER: str
    FILE_RETENTION_DAYS: int = 7
    MAX_FILE_SIZE: int = 1073741824  # bytes
    TUS_ENDPOINT: str
    SNIPPET_FORMAT: str = 'wav'

    # Models & authentication
    DEVICE: str = 'cpu'
    WHISPER_COMPUTE_TYPE: str = 'int8'
    WHISPER_MODEL: str            # e.g. "openai/whisper-large-v3"
    HUGGINGFACE_TOKEN: str
    PYANNOTE_PROTOCOL: str
    LANGUAGE_CODE: str = 'en'

    # Postgres
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    # SQLAlchemy / full DB URL
    DATABASE_URL: str

    # Redis (if your code uses REDIS_URL)
    REDIS_URL: str

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        # Use custom JSON loader for ALLOWED_ORIGINS
        json_loads = parse_origins

# Global settings instance
settings = Settings()