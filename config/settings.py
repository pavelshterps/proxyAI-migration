from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    UPLOAD_FOLDER: str = Field("/tmp/uploads", env="UPLOAD_FOLDER")
    CELERY_BROKER_URL: str = Field("redis://redis:6379", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field("redis://redis:6379", env="CELERY_RESULT_BACKEND")
    PYANNOTE_MODEL: str = Field("pyannote/speaker-diarization", env="PYANNOTE_MODEL")
    HF_TOKEN: str | None = Field(None, env="HF_TOKEN")
    WHISPER_MODEL: str = Field("openai/whisper-large-v2", env="WHISPER_MODEL")
    WHISPER_COMPUTE_TYPE: str = Field("float16", env="WHISPER_COMPUTE_TYPE")


settings = Settings()