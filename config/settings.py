# config/settings.py
import os

# папка с загруженными .wav
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
# куда сохранять результаты
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "/tmp/results")

# tusd
TUSD_ENDPOINT = os.getenv("TUSD_ENDPOINT", "http://tusd:1080/files/")

# Celery / Redis
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# Pyannote cache
DIARIZER_CACHE_DIR = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")

# Whisper
WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    "/hf_cache/models--guillaumekln--faster-whisper-medium"
)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE_INDEX = int(os.getenv("WHISPER_DEVICE_INDEX", 0))

# Concurrency
CPU_CONCURRENCY = int(os.getenv("CPU_CONCURRENCY", 1))
GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", 1))

# Pyannote model
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization")