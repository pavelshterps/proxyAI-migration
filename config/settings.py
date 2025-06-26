from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # пути и кэши
    UPLOAD_FOLDER: str
    HF_CACHE_DIR: str

    # брокер и бэкенд Celery
    BROKER_URL: str
    RESULT_BACKEND: str

    # модели
    PYANNOTE_MODEL: str
    WHISPER_MODEL: str

    # устройство и compute_type для whisper
    DEVICE_TYPE: str = "cuda"
    COMPUTE_TYPE: str = "float16"

    # конкарренси
    CPU_CONCURRENCY: int = 4
    GPU_CONCURRENCY: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()