import json
import logging
import subprocess
import time
from pathlib import Path
from celery.signals import worker_process_init
from redis import Redis

from config.settings import settings
from config.celery import celery_app
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

# Флаги наличия тяжёлых библиотек
_HF_AVAILABLE = False
_PN_AVAILABLE = False

# Ленивый импорт faster_whisper
try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info("[INIT] faster_whisper available")
except ImportError as e:
    logger.warning(f"[INIT] faster_whisper not available: {e}")

# Ленивый импорт pyannote.audio
try:
    from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
    _PN_AVAILABLE = True
    logger.info("[INIT] pyannote.audio available")
except ImportError as e:
    logger.warning(f"[INIT] pyannote.audio not available: {e}")


# === Lazy-инициализации ===

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    if not _HF_AVAILABLE:
        raise RuntimeError("WhisperModel unavailable")
    global _whisper_model
    if _whisper_model is None:
        model_id   = settings.WHISPER_MODEL_PATH
        cache_dir  = settings.HUGGINGFACE_CACHE_DIR
        device     = settings.WHISPER_DEVICE.lower()
        local_only = (device == "cpu")

        logger.info(f"[INIT] WhisperModel init: model={model_id}, device={device}, local_only={local_only}")
        try:
            model_path = download_model(model_id, cache_dir=cache_dir, local_files_only=local_only)
            logger.info(f"[INIT] Model cached at: {model_path}")
        except Exception as e:
            if local_only:
                logger.error(f"[INIT] Model missing from cache: {e}")
                raise
            logger.warning(f"[INIT] Will download '{model_id}' online: {e}")
            model_path = model_id

        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
        if device == "cpu" and compute in ("float16","fp16"):
            logger.warning("[INIT] FP16 unsupported on CPU, switching to int8")
            compute = "int8"

        _whisper_model = WhisperModel(model_path, device=device, compute_type=compute)
        logger.info("[INIT] WhisperModel loaded successfully")
    return _whisper_model

def get_vad():
    if not _PN_AVAILABLE:
        raise RuntimeError("VAD unavailable")
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
    if not _PN_AVAILABLE:
        raise RuntimeError("Diarizer unavailable")
    global _clustering_diarizer
    if _clustering_diarizer is None:
        logger.info("[INIT] Loading Diarizer model...")
        Path(settings.DIARIZER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("[INIT] Diarizer model loaded")
    return _clustering_diarizer


# === Warm-up воркера ===

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    sample = Path(__file__).parent / "tests/fixtures/sample.wav"
    try:
        logger.info("[WARMUP] Starting Whisper warm-up")
        get_whisper_model().transcribe(str(sample),
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}))
        logger.info("[WARMUP] Whisper warm-up completed")

        # только на GPU: инициализируем VAD+diarizer
        if settings.WHISPER_DEVICE.lower() != "cpu":
            logger.info("[WARMUP] Starting VAD + Diarizer warm-up")
            get_vad().apply({"audio": str(sample)})
            get_clustering_diarizer().apply({"audio": str(sample)})
            logger.info("[WARMUP] VAD + Diarizer warm-up completed")
    except Exception as e:
        logger.warning(f"[WARMUP] Failed: {e!r}")


# === Tasks ===

@celery_app.task(bind=True, name="tasks.download_audio", queue="transcribe_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    """noop-заглушка"""
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")


@celery_app.task(bind=True, name="tasks.preview_transcribe", queue="transcribe_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    """Превью-транскрипция первых секунд."""
    if not _HF_AVAILABLE:
        logger.error(f"[{correlation_id}] faster_whisper unavailable — пропускаем preview")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< PREVIEW START >>> for {upload_id}")

    # конверсия в WAV
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"), None)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(src, wav)
        logger.info(f"[{correlation_id}] Converted to WAV: {wav_path}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    # обрезка превью
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
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        segs, _ = model.transcribe(str(preview_wav), word_timestamps=True, **opts)
        segs = list(segs)
        logger.info(f"[{correlation_id}] Preview segments count: {len(segs)}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Preview transcription error: {e}")
        return

    preview = {
        "text": "".join(s.text for s in segs),
        "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    }
    # сохранить и опубликовать
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_file = out_dir / "preview.json"
    preview_file.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{correlation_id}] Preview JSON saved to {preview_file}")

    state = {"status": "preview_done", "preview": preview}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] <<< PREVIEW DONE >>> for {upload_id} in {time.time()-t0:.2f}s")

    # цепочка к полной транскрипции
    transcribe_segments.delay(upload_id, correlation_id)


@celery_app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    """Полная транскрипция всего файла."""
    if not _HF_AVAILABLE:
        logger.error(f"[{correlation_id}] faster_whisper unavailable — пропускаем full transcript")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< TRANSCRIBE START >>> for {upload_id}")

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        model = get_whisper_model()
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        segs, _ = model.transcribe(str(wav), word_timestamps=True, **opts)
        segs = list(segs)
        logger.info(f"[{correlation_id}] Full segments count: {len(segs)}")

        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = out_dir / "transcript.json"
        transcript_file.write_text(
            json.dumps([{"start":s.start,"end":s.end,"text":s.text} for s in segs],
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"[{correlation_id}] Transcript JSON saved to {transcript_file}")

        state = {"status": "transcript_done", "preview": None}
        r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        logger.info(f"[{correlation_id}] <<< TRANSCRIBE DONE >>> for {upload_id} in {time.time()-t0:.2f}s")

        # автозапуск диаризации, если флаг
        if r.get(f"diarize_requested:{upload_id}") == "1":
            diarize_full.delay(upload_id, correlation_id)
    except Exception as e:
        logger.error(f"[{correlation_id}] Error in full transcription: {e}", exc_info=True)


@celery_app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    """Диаризация спикеров."""
    if not _PN_AVAILABLE:
        logger.error(f"[{correlation_id}] pyannote.audio unavailable — пропускаем diarization")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< DIARIZE START >>> for {upload_id}")

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        ann = get_clustering_diarizer().apply({"audio": str(wav)})

        segs = [
            {"start": float(seg.start),
             "end": float(seg.end),
             "speaker": spk}
            for seg, _, spk in ann.itertracks(yield_label=True)
        ]

        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        diar_file = out_dir / "diarization.json"
        diar_file.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[{correlation_id}] Diarization JSON saved to {diar_file}")

        state = {"status":"diarization_done","preview":None}
        r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        logger.info(f"[{correlation_id}] <<< DIARIZE DONE >>> for {upload_id} in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.error(f"[{correlation_id}] Diarization error: {e}", exc_info=True)