# tasks.py
import os
import json
import logging
import requests
import subprocess
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
logger.setLevel(logging.INFO)  # убедимся, что INFO и выше выводятся

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    """
    Ленивая инициализация WhisperModel:
      1) download_model для кеширования (или разрешаем онлайн на GPU)
      2) передаём в конструктор только model_path, device, compute_type
    """
    global _whisper_model
    if _whisper_model is None:
        model_id   = settings.WHISPER_MODEL_PATH
        cache_dir  = settings.HUGGINGFACE_CACHE_DIR
        device     = settings.WHISPER_DEVICE.lower()
        local_only = (device == "cpu")

        logger.info(f"[INIT] WhisperModel init: model={model_id}, device={device}, cache_dir={cache_dir}, local_only={local_only}")
        try:
            model_path = download_model(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_only
            )
            logger.info(f"[INIT] Model cached at: {model_path}")
        except Exception as e:
            if local_only:
                logger.error(f"[INIT] Model '{model_id}' not in local cache and download disabled: {e}")
                raise
            logger.warning(f"[INIT] Will download '{model_id}' online: {e}")
            model_path = model_id

        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning("[INIT] CPU не поддерживает fp16, переключаем на int8")
            compute = "int8"

        _whisper_model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute
        )
        logger.info("[INIT] WhisperModel loaded successfully")
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        logger.info("[INIT] Loading VAD model...")
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("[INIT] VAD model loaded")
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        logger.info("[INIT] Loading Diarizer model...")
        os.makedirs(settings.DIARIZER_CACHE_DIR, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("[INIT] Diarizer model loaded")
    return _clustering_diarizer

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Warm-up при старте воркера
      - Прогрев WhisperModel на sample.wav
      - Если GPU — прогрев VAD + Diarizer
    """
    sample = Path(__file__).parent / "tests/fixtures/sample.wav"
    opts   = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    try:
        logger.info("[WARMUP] Starting Whisper warm-up")
        get_whisper_model().transcribe(str(sample), **opts)
        logger.info("[WARMUP] Whisper warm-up completed")
        if settings.WHISPER_DEVICE.lower() != "cpu":
            logger.info("[WARMUP] Starting VAD + Diarizer warm-up")
            get_vad().apply({"audio": str(sample)})
            get_clustering_diarizer().apply({"audio": str(sample)})
            logger.info("[WARMUP] VAD + Diarizer warm-up completed")
    except Exception as e:
        logger.warning(f"[WARMUP] Warm-up failed: {e!r}")

@celery_app.task(bind=True, name="tasks.download_audio", queue="transcribe_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")

@celery_app.task(bind=True, name="tasks.preview_transcribe", queue="transcribe_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    start_ts = time.time()
    logger.info(f"[{correlation_id}] <<< PREVIEW START >>> for {upload_id}")
    r   = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"), None)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    # конвертация
    try:
        wav_path = convert_to_wav(src, wav)
        logger.info(f"[{correlation_id}] Converted to WAV: {wav_path}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    # вырезаем предварительно первые N секунд
    preview_wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}_preview.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_path),
        "-t", str(settings.PREVIEW_LENGTH_S),
        str(preview_wav)
    ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"[{correlation_id}] Preview WAV generated: {preview_wav}")

    # транскрипция превью
    try:
        model = get_whisper_model()
    except Exception as e:
        logger.error(f"[{correlation_id}] Cannot load WhisperModel: {e}")
        return

    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    try:
        segs, _ = model.transcribe(str(preview_wav), word_timestamps=True, **opts)
        logger.info(f"[{correlation_id}] Whisper.transcribe(preview) returned {len(segs)} segments")
    except Exception as e:
        logger.error(f"[{correlation_id}] Preview transcription error: {e}")
        return

    preview = {"text": "", "timestamps": []}
    for s in segs:
        preview["text"] += s.text
        preview["timestamps"].append({
            "start": s.start, "end": s.end, "text": s.text
        })

    # сохраняем превью в Redis и логим
    r.set(f"preview_result:{upload_id}", json.dumps(preview, ensure_ascii=False))
    state = {"status": "preview_done", "preview": preview}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Preview done in {(time.time()-start_ts):.2f}s, data pushed to Redis")

    # (опционально) сохраняем локально файл с превью-транскриптом
    preview_file = Path(settings.RESULTS_FOLDER) / upload_id / "preview_transcript.json"
    preview_file.parent.mkdir(exist_ok=True, parents=True)
    preview_file.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{correlation_id}] Preview JSON saved to {preview_file}")

    # запускаем полную транскрипцию
    transcribe_segments.delay(upload_id, correlation_id)
    logger.info(f"[{correlation_id}] Launched full transcription task")

@celery_app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    start_ts = time.time()
    logger.info(f"[{correlation_id}] <<< TRANSCRIPT START >>> for {upload_id}")
    r   = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    try:
        model = get_whisper_model()
    except Exception as e:
        logger.error(f"[{correlation_id}] Cannot load WhisperModel: {e}")
        return

    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    try:
        segs, _ = model.transcribe(str(wav), word_timestamps=True, **opts)
        logger.info(f"[{correlation_id}] Whisper.transcribe(full) returned {len(segs)} segments")
    except Exception as e:
        logger.error(f"[{correlation_id}] Full transcription error: {e}")
        return

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = out_dir / "transcript.json"
    transcript_data = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    transcript_file.write_text(json.dumps(transcript_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{correlation_id}] Transcript saved to {transcript_file}")

    # публикуем статус в Redis
    state = {"status": "transcript_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Transcript done in {(time.time()-start_ts):.2f}s, status published")

    # отправляем внешние callbacks, если есть
    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb,
                          json={"event": "transcript_complete", "external_id": upload_id},
                          timeout=5)
            logger.info(f"[{correlation_id}] Webhook sent to {cb}")
        except Exception as e:
            logger.warning(f"[{correlation_id}] Webhook to {cb} failed: {e}")

    # автозапуск диаризации, если запрошено
    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)
        logger.info(f"[{correlation_id}] Launched diarization task (flag was set)")

@celery_app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    start_ts = time.time()
    logger.info(f"[{correlation_id}] <<< DIARIZATION START >>> for {upload_id}")
    r   = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
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

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    diar_file = out_dir / "diarization.json"
    diar_file.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{correlation_id}] Diarization saved to {diar_file}")

    state = {"status": "diarization_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] Diarization done in {(time.time()-start_ts):.2f}s, status published")

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb,
                          json={"event": "diarization_complete", "external_id": upload_id},
                          timeout=5)
            logger.info(f"[{correlation_id}] Diarization webhook sent to {cb}")
        except Exception as e:
            logger.warning(f"[{correlation_id}] Diarization webhook to {cb} failed: {e}")

@celery_app.task(name="tasks.cleanup_old_uploads", queue="cleanup")
def cleanup_old_uploads():
    cutoff = time.time() - settings.FILE_RETENTION_DAYS * 86400
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            logger.info(f"[CLEANUP] Removed old upload file {f}")
    logger.info("[CLEANUP] Old uploads cleanup completed")