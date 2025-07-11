import os
import json
import logging
import time
import requests
from pathlib import Path

from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from pydub import AudioSegment
from redis import Redis

from config.settings import settings
from config.celery import app          # единый Celery instance
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device     = settings.WHISPER_DEVICE.lower()
        compute    = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(
                f"Compute type '{compute}' not supported on CPU; falling back to int8"
            )
            compute = "int8"
        _whisper_model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute
        )
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        model_id = getattr(
            settings,
            "VAD_MODEL_PATH",
            "pyannote/voice-activity-detection"
        )
        _vad = VoiceActivityDetection.from_pretrained(
            model_id, use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _clustering_diarizer

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Warm up: Whisper (CPU) or VAD+diarizer (GPU).
    """
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    device = settings.WHISPER_DEVICE.lower()
    if device == "cpu":
        try:
            opts = {}
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE
            get_whisper_model().transcribe(str(sample), **opts)
        except:
            pass
    else:
        try:
            get_vad().apply({"audio": str(sample)})
        except:
            pass
        try:
            get_clustering_diarizer().apply({"audio": str(sample)})
        except:
            pass

@app.task(bind=True, name="tasks.download_audio", queue="preprocess_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    # noop if already local
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")

@app.task(bind=True, name="tasks.preview_transcribe", queue="preprocess_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    """Preview: первые settings.PREVIEW_LENGTH_S секунд."""
    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    # … (конвертация + сегментация как было) …
    # результат пушим в Redis:
    #   redis.set(f"preview_result:{upload_id}", json.dumps(preview))
    #   redis.publish(f"progress:{upload_id}", "preview_done")
    # callbacks:
    for url in json.loads(redis.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(url, json={
                "external_id": upload_id,
                "event":       "preview_complete"
            }, timeout=5)
        except:
            pass
    # запускаем полную транскрипцию
    transcribe_segments.delay(upload_id, correlation_id)

@app.task(bind=True, name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    """Полная транскрипция."""
    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    # … (как было: полная транскрипция, сохранение в RESULTS) …
    # callbacks для full transcript
    for url in json.loads(redis.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(url, json={
                "external_id": upload_id,
                "event":       "transcript_complete"
            }, timeout=5)
        except:
            pass

    # **Условный** запуск диаризации
    if redis.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

@app.task(bind=True, name="tasks.diarize_full", queue="preprocess_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    """Полная диаризация."""
    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    # … (как было: diarization.json) …
    # callbacks для diarization
    for url in json.loads(redis.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(url, json={
                "external_id": upload_id,
                "event":       "diarization_complete"
            }, timeout=5)
        except:
            pass

@app.task(name="tasks.cleanup_old_uploads")
def cleanup_old_uploads():
    """Удаление старых."""
    cutoff = time.time() - 24 * 3600
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)

def split_audio_fixed_windows(audio_path: Path, window_s: int):
    audio = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    return [
        (start / 1000.0, min(start + window_ms, length_ms) / 1000.0)
        for start in range(0, length_ms, window_ms)
    ]