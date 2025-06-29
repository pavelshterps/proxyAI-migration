# config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # FastAPI
    fastapi_host: str = Field("0.0.0.0", env="FASTAPI_HOST")
    fastapi_port: int = Field(8000,       env="FASTAPI_PORT")
    api_workers:  int = Field(1,          env="API_WORKERS")
    allowed_origins: list[str] = Field(
        ["*"], env="ALLOWED_ORIGINS", description="CORS origins"
    )

    # Celery / Redis
    celery_broker_url:    str = Field(..., env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(..., env="CELERY_RESULT_BACKEND")
    cpu_concurrency:       int = Field(4,  env="CPU_CONCURRENCY")
    gpu_concurrency:       int = Field(1,  env="GPU_CONCURRENCY")
    timezone:             str = Field("UTC", env="TIMEZONE")

    # Storage
    upload_folder:      str = Field("/tmp/uploads",   env="UPLOAD_FOLDER")
    results_folder:     str = Field("/tmp/results",   env="RESULTS_FOLDER")
    file_retention_days: int = Field(7,                env="FILE_RETENTION_DAYS")
    max_file_size:       int = Field(1<<30,            env="MAX_FILE_SIZE")

    # tusd
    tusd_endpoint: str = Field(..., env="TUSD_ENDPOINT")
    snippet_format: str = Field("wav", env="SNIPPET_FORMAT")

    # Pyannote diarizer
    diarizer_cache_dir: str = Field("/tmp/diarizer_cache", env="DIARIZER_CACHE_DIR")
    pyannote_protocol:  str = Field("pyannote/speaker-diarization", env="PYANNOTE_PROTOCOL")

    # Hugging Face
    huggingface_token: str = Field(..., env="HUGGINGFACE_TOKEN")
    hf_cache_dir:      str = Field("/hf_cache", env="HF_CACHE_DIR")

    # Whisper / Faster‐Whisper
    whisper_model_path:  str = Field(..., env="WHISPER_MODEL_PATH")
    whisper_device:      str = Field("cuda",  env="WHISPER_DEVICE")
    whisper_device_index: int = Field(0,      env="WHISPER_DEVICE_INDEX")
    whisper_compute_type: str = Field("int8", env="WHISPER_COMPUTE_TYPE")
    whisper_beam_size:    int = Field(5,     env="WHISPER_BEAM_SIZE")
    whisper_task:         str = Field("transcribe", env="WHISPER_TASK")
    segment_length_s:     int = Field(30,    env="SEGMENT_LENGTH_S")

    # Cleanup
    clean_up_uploads: bool = Field(True, env="CLEAN_UP_UPLOADS")

    # Database / Misc
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url:    str = Field(..., env="REDIS_URL")

    # Pydantic config
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"   # drop any env vars you didn't declare
    )

settings = Settings()