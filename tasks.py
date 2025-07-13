# tasks.py
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
    - cache_dir берём из env: /hf_cache
    - local_files_only=True для CPU, False для GPU (чтобы при необходимости скачать)
    """
    global _whisper_model
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        cache_dir = settings.HUGGINGFACE_CACHE_DIR
        device = settings.WHISPER_DEVICE.lower()
        # на CPU не можем скачивать онлайн, на GPU — можем
        local_only = (device == "cpu")

        logger.info(f"Initializing WhisperModel: model={model_id}, cache_dir={cache_dir}, "
                    f"device={device}, local_only={local_only}")

        # Проверяем, есть ли модель в кеше
        try:
            download_model(model_id, cache_dir=cache_dir, local_files_only=local_only)
            logger.info(f"Whisper model '{model_id}' found in cache")
        except Exception as e:
            if local_only:
                # на CPU без онлайн-загрузки — это ошибка
                logger.error(f"Model '{model_id}' missing from cache and download disabled: {e}")
                raise RuntimeError(f"Whisper model '{model_id}' not in local cache '{cache_dir}'")
            else:
                # на GPU — просто залогируем, дальше модель сама её скачает
                logger.warning(f"Model '{model_id}' not in cache, will download online: {e}")

        # Выбираем compute_type, на CPU запрещаем float16/fp16
        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning("FP16 не поддерживается на CPU, переключаем на int8")
            compute = "int8"

        # Создаём модель; на GPU local_files_only=False по умолчанию
        _whisper_model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute,
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
        logger.info("WhisperModel loaded successfully")

    return _whisper_model

def get_vad():
    """
    Ленивая инициализация VAD-модели (pyannote.audio).
    """
    global _vad
    if _vad is None:
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("VAD model loaded")
    return _vad

def get_clustering_diarizer():
    """
    Ленивая инициализация SpeakerDiarization (pyannote.audio).
    """
    global _clustering_diarizer
    if _clustering_diarizer is None:
        os.makedirs(settings.DIARIZER_CACHE_DIR, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("Diarizer model loaded")
    return _clustering_diarizer

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Warm-up моделей при старте воркера:
    - WhisperModel.transcribe на небольшой sample.wav под любым устройством
    - VAD и Diarizer для GPU-режима
    """
    sample = Path(__file__).parent / "tests/fixtures/sample.wav"
    try:
        # всегда прогреваем WhisperModel (и на GPU, и на CPU)
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        get_whisper_model().transcribe(str(sample), **opts)
        logger.info("Whisper warm-up completed")
        # если GPU — дополнительно прогреваем VAD + Diarizer
        if settings.WHISPER_DEVICE.lower() != "cpu":
            get_vad().apply({"audio": str(sample)})
            get_clustering_diarizer().apply({"audio": str(sample)})
            logger.info("VAD+Diarizer warm-up completed")
    except Exception as e:
        # не фейлим воркер — просто логируем
        logger.warning(f"Warm-up failed: {e!r}")

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
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    try:
        model = get_whisper_model()
    except Exception as e:
        logger.error(f"[{correlation_id}] Cannot load WhisperModel: {e}")
        return

    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    try:
        segs, _ = model.transcribe(str(wav_path), word_timestamps=True, **opts)
    except Exception as e:
        logger.error(f"[{correlation_id}] Preview transcription error: {e}")
        return

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
    logger.info(f"[{correlation_id}] Preview done for {upload_id}")

    transcribe_segments.delay(upload_id, correlation_id)

@celery_app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    try:
        model = get_whisper_model()
    except Exception as e:
        logger.error(f"[{correlation_id}] Cannot load WhisperModel: {e}")
        return

    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    try:
        segs, _ = model.transcribe(str(wav), word_timestamps=True, **opts)
    except Exception as e:
        logger.error(f"[{correlation_id}] Full transcription error: {e}")
        return

    out = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    state = {"status": "transcript_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Full transcription done for {upload_id}")

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb,
                json={"event": "transcript_complete", "external_id": upload_id},
                timeout=5
            )
        except Exception:
            pass

    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

@celery_app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    try:
        ann = get_clustering_diarizer().apply({"audio": str(wav)})
    except Exception as e:
        logger.error(f"[{correlation_id}] Diarization error: {e}")
        return

    segs = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": spk}
        for seg, _, spk in ann.itertracks(yield_label=True)
    ]

    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "diarization.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    state = {"status": "diarization_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Diarization done for {upload_id}")

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb,
                json={"event":"diarization_complete","external_id":upload_id},
                timeout=5
            )
        except Exception:
            pass

@celery_app.task(name="tasks.cleanup_old_uploads", queue="cleanup")
def cleanup_old_uploads():
    """
    Удаляем файлы старше FILE_RETENTION_DAYS.
    """
    cutoff = time.time() - settings.FILE_RETENTION_DAYS * 86400
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
    logger.info("Old uploads cleaned up")