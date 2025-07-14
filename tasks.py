import json
import logging
import time
from pathlib import Path
from celery.signals import worker_process_init
from redis import Redis

from config.settings import settings
from config.celery import celery_app
from utils.audio import convert_to_wav, download_audio

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info("[INIT] faster_whisper available")
except ImportError as e:
    _HF_AVAILABLE = False
    logger.warning(f"[INIT] faster_whisper not available: {e}")

try:
    from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
    _PN_AVAILABLE = True
    logger.info("[INIT] pyannote.audio available")
except ImportError as e:
    _PN_AVAILABLE = False
    logger.warning(f"[INIT] pyannote.audio not available: {e}")

_whisper_model = _vad = _clustering_diarizer = None

def get_whisper_model():
    global _whisper_model
    # (без изменений)
    model_id = settings.WHISPER_MODEL_PATH
    cache = settings.HUGGINGFACE_CACHE_DIR
    device = settings.WHISPER_DEVICE.lower()
    local = (device == "cpu")
    try:
        path = download_model(model_id, cache_dir=cache, local_files_only=local)
    except Exception:
        path = model_id
    compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
    if device == "cpu" and compute in ("fp16", "float16"):
        compute = "int8"
    _whisper_model = WhisperModel(path, device=device, compute_type=compute)
    return _whisper_model

def get_vad():
    global _vad
    # (без изменений)
    _vad = VoiceActivityDetection.from_pretrained(
        settings.VAD_MODEL_PATH,
        cache_dir=settings.HUGGINGFACE_CACHE_DIR,
        use_auth_token=settings.HUGGINGFACE_TOKEN
    )
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    # (без изменений)
    Path(settings.DIARIZER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    _clustering_diarizer = SpeakerDiarization.from_pretrained(
        settings.PYANNOTE_PIPELINE,
        cache_dir=settings.DIARIZER_CACHE_DIR,
        use_auth_token=settings.HUGGINGFACE_TOKEN
    )
    return _clustering_diarizer

@worker_process_init.connect
def preload_on_startup(**kwargs):
    if _HF_AVAILABLE:
        sample = Path(__file__).parent / "tests/fixtures/sample.wav"
        try:
            get_whisper_model().transcribe(
                str(sample), max_initial_timestamp=settings.PREVIEW_LENGTH_S
            )
        except Exception:
            logger.warning("[WARMUP] Whisper warm-up failed")
    if _PN_AVAILABLE and settings.WHISPER_DEVICE.lower().startswith("cuda"):
        try:
            get_vad()
            get_clustering_diarizer()
        except Exception:
            logger.warning("[WARMUP] VAD/Diarizer warm-up failed")

@celery_app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id, url: str = None):
    cid = correlation_id or "?"
    if not _HF_AVAILABLE:
        logger.error(f"[{cid}] no Whisper available")
        return
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    # скачиваем, если URL задан
    if url:
        src = download_audio(url, Path(settings.UPLOAD_FOLDER) / f"{upload_id}_src")
    else:
        src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"), None)

    if src is None or not src.exists():
        err = f"source not found for upload_id={upload_id}"
        logger.error(f"[{cid}] {err}")
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": err}))
        return

    try:
        wav = convert_to_wav(src, Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav")
        segs, _ = get_whisper_model().transcribe(
            str(wav), word_timestamps=True,
            max_initial_timestamp=settings.PREVIEW_LENGTH_S,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
    except Exception as e:
        logger.error(f"[{cid}] preview error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        return

    segs = list(segs)
    preview = {
        "text": "".join(s.text for s in segs),
        "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    }

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(exist_ok=True)
    (out / "preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    r.publish(f"progress:{upload_id}", json.dumps({"status": "preview_done", "preview": preview}))
    transcribe_segments.delay(upload_id, correlation_id)
    logger.info(f"[{cid}] PREVIEW done in {time.time() - t0:.2f}s")

@celery_app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    if not _HF_AVAILABLE:
        logger.error(f"[{cid}] no Whisper available")
        return
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav.exists():
        err = f"wav file not found for upload_id={upload_id}"
        logger.error(f"[{cid}] {err}")
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": err}))
        return

    try:
        segs, _ = get_whisper_model().transcribe(
            str(wav), word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
    except Exception as e:
        logger.error(f"[{cid}] transcribe error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        return

    segs = list(segs)
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(exist_ok=True)
    (out / "transcript.json").write_text(
        json.dumps([{"start": s.start, "end": s.end, "text": s.text} for s in segs],
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))

    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

    logger.info(f"[{cid}] TRANSCRIBE done in {time.time() - t0:.2f}s")

@celery_app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    if not _PN_AVAILABLE:
        logger.error(f"[{cid}] no pyannote available")
        return
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav.exists():
        err = f"wav file not found for upload_id={upload_id}"
        logger.error(f"[{cid}] {err}")
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": err}))
        return

    try:
        get_vad()
        get_clustering_diarizer()
        ann = get_clustering_diarizer().apply({"audio": str(wav)})
    except Exception as e:
        logger.error(f"[{cid}] diarize error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        return

    segs = [{"start": float(s.start), "end": float(s.end), "speaker": spk}
            for s, _, spk in ann.itertracks(yield_label=True)]
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(exist_ok=True)
    (out / "diarization.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarization_done"}))
    logger.info(f"[{cid}] DIARIZE done in {time.time() - t0:.2f}s")