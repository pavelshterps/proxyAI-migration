# tasks.py
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

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    """
    Lazy init of WhisperModel:
      1) download_model for cache prefetching (or allow online on GPU)
      2) pass only model_path, device, compute_type to WhisperModel
    """
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
                raise RuntimeError(f"Model '{model_id}' not in local cache")
            logger.warning(f"[INIT] Will download '{model_id}' online at init: {e}")
            model_path = model_id

        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
        if device == "cpu" and compute in ("float16","fp16"):
            logger.warning("[INIT] FP16 unsupported on CPU, switching to int8")
            compute = "int8"

        _whisper_model = WhisperModel(model_path, device=device, compute_type=compute)
        logger.info("[INIT] WhisperModel loaded successfully")
    return _whisper_model

def get_vad():
    """Lazy init of Voice Activity Detection."""
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
    """Lazy init of Speaker Diarization."""
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

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Worker warm-up on startup:
      • Warm up WhisperModel on a small sample.wav
      • If GPU: also warm up VAD + Diarizer
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
        logger.warning(f"[WARMUP] Failed: {e!r}")

@celery_app.task(bind=True, name="tasks.download_audio", queue="transcribe_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")

@celery_app.task(bind=True, name="tasks.preview_transcribe", queue="transcribe_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    """
    1) convert uploaded file to WAV
    2) cut first PREVIEW_LENGTH_S seconds to a _preview.wav
    3) run WhisperModel.transcribe on that small chunk
    4) publish to Redis + write preview.json to disk
    """
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< PREVIEW START >>> for {upload_id}")

    # --- 1) convert
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"), None)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(src, wav)
        logger.info(f"[{correlation_id}] Converted to WAV: {wav_path}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    # --- 2) extract preview chunk
    preview_wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}_preview.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-t", str(settings.PREVIEW_LENGTH_S),
        str(preview_wav)
    ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"[{correlation_id}] Preview WAV generated: {preview_wav}")

    # --- 3) transcribe preview
    try:
        model = get_whisper_model()
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        segs, _meta = model.transcribe(str(preview_wav), word_timestamps=True, **opts)
        segs = list(segs)
        logger.info(f"[{correlation_id}] Preview segments count: {len(segs)}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Preview transcription error: {e}")
        return

    preview = {"text":"".join(s.text for s in segs),
               "timestamps":[{"start":s.start,"end":s.end,"text":s.text} for s in segs]}

    # --- write preview.json to RESULTS_FOLDER
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_file = out_dir / "preview.json"
    preview_file.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{correlation_id}] Preview JSON saved to {preview_file}")

    # --- publish to Redis SSE
    state = {"status":"preview_done","preview":preview}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    t1 = time.time()
    logger.info(f"[{correlation_id}] <<< PREVIEW DONE >>> for {upload_id} in {t1-t0:.2f}s")

    # --- 4) chain to full transcription
    transcribe_segments.delay(upload_id, correlation_id)

@celery_app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    """
    Full transcription on the entire WAV, write transcript.json, publish SSE.
    """
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< TRANSCRIBE START >>> for {upload_id}")

    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    try:
        model = get_whisper_model()
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        segs, _meta = model.transcribe(str(wav), word_timestamps=True, **opts)
        segs = list(segs)
        logger.info(f"[{correlation_id}] Full segments count: {len(segs)}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Full transcription error: {e}")
        return

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = out_dir / "transcript.json"
    transcript_file.write_text(
        json.dumps([{"start":s.start,"end":s.end,"text":s.text} for s in segs],
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[{correlation_id}] Transcript JSON saved to {transcript_file}")

    state = {"status":"transcript_done","preview":None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    t1 = time.time()
    logger.info(f"[{correlation_id}] <<< TRANSCRIBE DONE >>> for {upload_id} in {t1-t0:.2f}s")

    # callbacks if any
    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb, json={"event":"transcript_complete","external_id":upload_id}, timeout=5)
            logger.info(f"[{correlation_id}] Callback sent to {cb}")
        except Exception:
            logger.warning(f"[{correlation_id}] Callback failed to {cb}")

    # chain to diarization if requested
    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

@celery_app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    """
    Speaker diarization, write diarization.json, publish SSE.
    """
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< DIARIZE START >>> for {upload_id}")

    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    try:
        ann = get_clustering_diarizer().apply({"audio": str(wav)})
    except Exception as e:
        logger.error(f"[{correlation_id}] Diarization error: {e}")
        return

    segs = [{"start":float(seg.start),"end":float(seg.end),"speaker":spk}
            for seg,_,spk in ann.itertracks(yield_label=True)]

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    diar_file = out_dir / "diarization.json"
    diar_file.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[{correlation_id}] Diarization JSON saved to {diar_file}")

    state = {"status":"diarization_done","preview":None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    t1 = time.time()
    logger.info(f"[{correlation_id}] <<< DIARIZE DONE >>> for {upload_id} in {t1-t0:.2f}s")

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb, json={"event":"diarization_complete","external_id":upload_id}, timeout=5)
            logger.info(f"[{correlation_id}] Callback sent to {cb}")
        except Exception:
            logger.warning(f"[{correlation_id}] Callback failed to {cb}")

@celery_app.task(name="tasks.cleanup_old_uploads", queue="cleanup")
def cleanup_old_uploads():
    cutoff = time.time() - settings.FILE_RETENTION_DAYS * 86400
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
    logger.info("[CLEANUP] Old uploads cleaned up")