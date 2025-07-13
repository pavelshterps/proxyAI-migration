import os
import json
import logging
import requests
import time
from pathlib import Path
from celery.signals import worker_process_init
from faster_whisper import WhisperModel, download_model
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from redis import Redis

from config.settings import settings
from config.celery import celery_app
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    """
    Ленивая инициализация WhisperModel.
    Проверяем кеш в /hf_cache, при необходимости скачиваем.
    """
    global _whisper_model
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        cache = settings.HUGGINGFACE_CACHE_DIR
        logger.info(f"WhisperModel init: model={model_id}, cache_dir={cache}")

        try:
            download_model(model_id, cache_dir=cache, local_files_only=True)
            logger.info(f"Model '{model_id}' найдена локально в кеше")
        except Exception:
            logger.info(f"Модель '{model_id}' не найдена в кеше — будет скачана из HuggingFace")

        device = settings.WHISPER_DEVICE.lower()
        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(f"FP16 не поддерживается на CPU — переключаемся на int8")
            compute = "int8"

        _whisper_model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute,
            cache_dir=cache,
            local_files_only=False  # разрешаем скачивание, если нужно
        )
        logger.info("WhisperModel успешно загружена")

    return _whisper_model

def get_vad():
    """
    Ленивая инициализация VAD (pyannote).
    """
    global _vad
    if _vad is None:
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("VAD модель загружена")
    return _vad

def get_clustering_diarizer():
    """
    Ленивая инициализация Speaker Diarization (pyannote).
    """
    global _clustering_diarizer
    if _clustering_diarizer is None:
        cache = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("Diarizer модель загружена")
    return _clustering_diarizer

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Прогрев моделей на старте worker-процесса.
    """
    sample = Path(__file__).parent / "tests/fixtures/sample.wav"
    try:
        if settings.WHISPER_DEVICE.lower() == "cpu":
            opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
            get_whisper_model().transcribe(str(sample), **opts)
        else:
            get_vad().apply({"audio": str(sample)})
            get_clustering_diarizer().apply({"audio": str(sample)})
        logger.info("Warm-up моделей выполнен успешно")
    except Exception as e:
        logger.warning("Warm-up моделей не удался: %r", e)

@celery_app.task(bind=True, name="tasks.download_audio", queue="transcribe_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")

@celery_app.task(bind=True, name="tasks.preview_transcribe", queue="transcribe_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"), None)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    try:
        wav_path = convert_to_wav(src, wav)
    except Exception as e:
        logger.error(f"[{correlation_id}] Ошибка конверсии: {e}")
        return

    model = get_whisper_model()
    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segs, _ = model.transcribe(str(wav_path), word_timestamps=True, **opts)

    preview = {"text": "", "timestamps": []}
    for s in segs:
        if s.start >= settings.PREVIEW_LENGTH_S:
            break
        preview["text"] += s.text
        preview["timestamps"].append({"start": s.start, "end": s.end, "text": s.text})

    r.set(f"preview_result:{upload_id}", json.dumps(preview, ensure_ascii=False))
    state = {"status": "preview_done", "preview": preview}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Preview транскрибирован для {upload_id}")

    transcribe_segments.delay(upload_id, correlation_id)

@celery_app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    model = get_whisper_model()
    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segs, _ = model.transcribe(str(wav), word_timestamps=True, **opts)

    out = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(exist_ok=True, parents=True)
    (d / "transcript.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    state = {"status": "transcript_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Полная транскрипция завершена для {upload_id}")

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb, json={"event": "transcript_complete","external_id":upload_id}, timeout=5)
        except Exception:
            pass

    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

@celery_app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    ann = get_clustering_diarizer().apply({"audio": str(wav)})
    segs = [{"start": float(seg.start), "end": float(seg.end), "speaker": spk}
            for seg, _, spk in ann.itertracks(yield_label=True)]

    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(exist_ok=True, parents=True)
    (d / "diarization.json").write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")

    state = {"status": "diarization_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Диаризация завершена для {upload_id}")

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb, json={"event":"diarization_complete","external_id":upload_id}, timeout=5)
        except Exception:
            pass

@celery_app.task(name="tasks.cleanup_old_uploads", queue="cleanup")
def cleanup_old_uploads():
    cutoff = time.time() - settings.FILE_RETENTION_DAYS * 86400
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
    logger.info("Очищены старые загрузки")