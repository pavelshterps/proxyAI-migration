import os
import logging
from celery import Celery
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)
app = Celery("proxyai")
app.config_from_object("config")

# Путь для кеша диаризатора
DIARIZER_CACHE = os.environ.get("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")

# Глобальные одноразовые объекты
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = "/hf_cache/models--guillaumekln--faster-whisper-medium"
        logger.info(f"Loading WhisperModel once at startup: "
                    f"{{'model_size_or_path': model_path, 'device': 'cuda', 'compute_type': 'int8', 'device_index': 0}}")
        _whisper_model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="int8",
            device_index=0,
        )
        logger.info("WhisperModel loaded (quantized int8)")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        # убеждаемся, что кеш-каталог есть и записываем в него
        os.makedirs(DIARIZER_CACHE, exist_ok=True)
        logger.info(f"Loading Pyannote diarizer with cache at {DIARIZER_CACHE}")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=True,
            cache_dir=DIARIZER_CACHE
        )
        logger.info("Pyannote diarizer loaded")
    return _diarizer


@app.task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id, segments):
    model = get_whisper_model()
    # ... остальной код транскрипции ...


@app.task(name="tasks.diarize_full")
def diarize_full(upload_id, audio_path):
    diarizer = get_diarizer()
    # ... остальной код диаризации ...