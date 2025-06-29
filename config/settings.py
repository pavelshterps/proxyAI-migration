from pydantic import BaseSettings

class Settings(BaseSettings):
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    TIMEZONE: str = "UTC"
    UPLOAD_FOLDER: str
    RESULTS_FOLDER: str

    class Config:
        env_file = ".env"

settings = Settings()