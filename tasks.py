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

# — флаги доступности моделей и логирование их наличия —
try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info("[INIT] faster_whisper library is available")  # LOG
except ImportError as e:
    _HF_AVAILABLE = False
    logger.warning(f"[INIT] faster_whisper not available: {e}")  # LOG

try:
    from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
    _PN_AVAILABLE = True
    logger.info("[INIT] pyannote.audio library is available")  # LOG
except ImportError as e:
    _PN_AVAILABLE = False
    logger.warning(f"[INIT] pyannote.audio not available: {e}")  # LOG

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("[WHISPER] Initializing WhisperModel...")  # LOG
        model_id = settings.WHISPER_MODEL_PATH
        cache = settings.HUGGINGFACE_CACHE_DIR
        device = settings.WHISPER_DEVICE.lower()
        local = (device == "cpu")
        try:
            path = download_model(model_id, cache_dir=cache, local_files_only=local)
            logger.info(f"[WHISPER] Model downloaded/cached at {path}")  # LOG
        except Exception:
            path = model_id
            logger.info(f"[WHISPER] Using local model path {path}")  # LOG
        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
        if device == "cpu" and compute in ("fp16", "float16"):
            compute = "int8"
            logger.info(f"[WHISPER] Forced compute_type to int8 on CPU")  # LOG
        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info(f"[WHISPER] WhisperModel initialized on {device} with compute_type={compute}")  # LOG
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        logger.info("[VAD] Loading VoiceActivityDetection model...")  # LOG
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"[VAD] VoiceActivityDetection loaded from {settings.VAD_MODEL_PATH}")  # LOG
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        logger.info("[DIARIZER] Loading SpeakerDiarization pipeline...")  # LOG
        Path(settings.DIARIZER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"[DIARIZER] SpeakerDiarization loaded from {settings.PYANNOTE_PIPELINE}")  # LOG
    return _clustering_diarizer

@worker_process_init.connect
def preload_on_startup(**kwargs):
    logger.info("[WARMUP] Worker process init — starting model warmup")  # LOG
    if _HF_AVAILABLE:
        sample = Path(__file__).parent / "tests/fixtures/sample.wav"
        try:
            logger.info("[WARMUP] Warming up WhisperModel with sample audio")  # LOG
            get_whisper_model().transcribe(
                str(sample),
                max_initial_timestamp=settings.PREVIEW_LENGTH_S
            )
            logger.info("[WARMUP] WhisperModel warmup succeeded")  # LOG
        except Exception:
            logger.warning("[WARMUP] WhisperModel warmup failed")  # LOG
    if _PN_AVAILABLE and settings.WHISPER_DEVICE.lower().startswith("cuda"):
        try:
            logger.info("[WARMUP] Warming up VAD & Diarizer on GPU")  # LOG
            get_vad()
            get_clustering_diarizer()
            logger.info("[WARMUP] VAD & Diarizer warmup succeeded")  # LOG
        except Exception:
            logger.warning("[WARMUP] VAD/Diarizer warmup failed")  # LOG

@celery_app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id, url: str):
    """
    1) Скачиваем аудио по URL
    2) Конвертируем в WAV
    3) Делаем превью-транскрибацию (WhisperModel.max_initial_timestamp)
    """
    cid = correlation_id or "?"
    logger.info(f"[{cid}] PREVIEW task started for upload_id={upload_id}")  # LOG
    if not _HF_AVAILABLE:
        logger.error(f"[{cid}] no Whisper available, aborting preview")  # LOG
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    try:
        logger.info(f"[{cid}] Downloading audio from URL: {url}")  # LOG
        src = download_audio(url, Path(settings.UPLOAD_FOLDER), upload_id)
        logger.info(f"[{cid}] Audio downloaded to {src}")  # LOG

        logger.info(f"[{cid}] Converting to WAV")  # LOG
        wav = convert_to_wav(src, Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav")
        logger.info(f"[{cid}] WAV file at {wav}")  # LOG

        logger.info(f"[{cid}] Running Whisper preview transcription")  # LOG
        segs, _ = get_whisper_model().transcribe(
            str(wav),
            word_timestamps=True,
            max_initial_timestamp=settings.PREVIEW_LENGTH_S,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        logger.info(f"[{cid}] Whisper preview returned {len(list(segs))} segments")  # LOG
    except Exception as e:
        logger.error(f"[{cid}] preview error: {e}", exc_info=True)  # LOG
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
    logger.info(f"[{cid}] Preview JSON written to {out/'preview.json'}")  # LOG

    r.publish(f"progress:{upload_id}", json.dumps({"status": "preview_done", "preview": preview}))
    transcribe_segments.delay(upload_id, correlation_id)
    logger.info(f"[{cid}] PREVIEW done in {time.time() - t0:.2f}s")  # LOG

@celery_app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{cid}] TRANSCRIBE task started for upload_id={upload_id}")  # LOG
    if not _HF_AVAILABLE:
        logger.error(f"[{cid}] no Whisper available, aborting full transcription")  # LOG
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        logger.info(f"[{cid}] Running Whisper full transcription on {wav}")  # LOG
        segs, _ = get_whisper_model().transcribe(
            str(wav),
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        logger.info(f"[{cid}] Whisper transcript returned {len(list(segs))} segments")  # LOG
    except Exception as e:
        logger.error(f"[{cid}] transcribe error: {e}", exc_info=True)  # LOG
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
    logger.info(f"[{cid}] Transcript JSON written to {out/'transcript.json'}")  # LOG

    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))
    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)
        logger.info(f"[{cid}] Auto-diarize triggered")  # LOG

    logger.info(f"[{cid}] TRANSCRIBE done in {time.time() - t0:.2f}s")  # LOG

@celery_app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{cid}] DIARIZE task started for upload_id={upload_id}")  # LOG
    if not _PN_AVAILABLE:
        logger.error(f"[{cid}] no pyannote available, aborting diarization")  # LOG
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        logger.info(f"[{cid}] Running VAD & Diarizer on {wav}")  # LOG
        get_vad()
        get_clustering_diarizer()
        ann = get_clustering_diarizer().apply({"audio": str(wav)})
        logger.info(f"[{cid}] Diarizer returned {len(list(ann.itertracks(yield_label=True)))} segments")  # LOG
    except Exception as e:
        logger.error(f"[{cid}] diarize error: {e}", exc_info=True)  # LOG
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        return

    segs = [
        {"start": float(s.start), "end": float(s.end), "speaker": spk}
        for s, _, spk in ann.itertracks(yield_label=True)
    ]
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(exist_ok=True)
    (out / "diarization.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[{cid}] Diarization JSON written to {out/'diarization.json'}")  # LOG

    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarization_done"}))
    logger.info(f"[{cid}] DIARIZE done in {time.time() - t0:.2f}s")  # LOG