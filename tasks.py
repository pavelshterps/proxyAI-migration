import os
from celery import Celery
from config.settings import PYANNOTE_PROTOCOL, HUGGINGFACE_TOKEN, UPLOAD_FOLDER
from faster_whisper import WhisperModel
from config.settings import DEVICE, WHISPER_MODEL, WHISPER_COMPUTE_TYPE, ALIGN_MODEL_NAME, ALIGN_BEAM_SIZE
from config.settings import TUS_ENDPOINT, MAX_FILE_SIZE_MB, SNIPPET_FORMAT

# Celery app is defined in celery_app.py (imported here just for context)
from celery_app import app

# Keep Whisper model global so we don't reload it on every segment
_whisper_model = None
_diarizer = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        # Mount a volume at /hf_cache in docker-compose
        cache_dir = os.getenv("HF_CACHE_DIR", "/hf_cache/pyannote")
        os.makedirs(cache_dir, exist_ok=True)
        _diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            cache_dir=cache_dir,
            use_auth_token=HUGGINGFACE_TOKEN
        )
    return _diarizer

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            device_index=0,
            compute_type=WHISPER_COMPUTE_TYPE,
            inter_threads=1,
            intra_threads=1
        )
    return _whisper_model

@app.task(name="tasks.transcribe_full")
def transcribe_full(filepath):
    # 1) Diarize
    diarizer = get_diarizer()
    diarization = diarizer(filepath)

    # Write snippet‐by-speaker, etc. (omitted)
    # 2) Chunked Whisper
    whisper = get_whisper_model()
    segments, _ = whisper.transcribe(filepath, beam_size=ALIGN_BEAM_SIZE)

    # Return segments (or align them)
    return list(segments)