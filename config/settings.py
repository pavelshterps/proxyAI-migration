from pydantic import BaseSettings


class Settings(BaseSettings):
    # Celery
    celery_broker_url: str
    celery_result_backend: str

    # API / workers
    api_workers: int = 1
    cpu_concurrency: int = 1
    gpu_concurrency: int = 1

    # Paths
    upload_folder: str = "/tmp/uploads"
    results_folder: str = "/tmp/results"
    whisper_model_path: str = "/hf_cache/models--guillaumekln--faster-whisper-medium"
    diarizer_cache_dir: str = "/tmp/diarizer_cache"

    # Whisper settings
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8"

    # Segmentation / VAD
    segment_length_s: int = 30

    # Timezone for Celery
    timezone: str = "UTC"

    # Flower auth
    flower_user: str = ""
    flower_pass: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()