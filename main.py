import os
import json
import logging
import requests
import time
from pathlib import Path

from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from redis import Redis

from config.settings import settings
from config.celery import app
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        device  = settings.WHISPER_DEVICE.lower()
        compute = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(f"Compute '{compute}' unsupported on CPU; using int8")
            compute = "int8"
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=device,
            compute_type=compute
        )
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        _vad = VoiceActivityDetection.from_pretrained(
            getattr(settings, "VAD_MODEL_PATH", "pyannote/voice-activity-detection"),
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        cache = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _clustering_diarizer

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    sample = Path(__file__).parent / "tests" / "fixtures" / "sample.wav"
    device = settings.WHISPER_DEVICE.lower()
    if device == "cpu":
        try:
            opts = {}
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE
            get_whisper_model().transcribe(str(sample), **opts)
        except: pass
    else:
        try: get_vad().apply({"audio": str(sample)})
        except: pass
        try: get_clustering_diarizer().apply({"audio": str(sample)})
        except: pass

# ===== все задачи ставятся теперь в общую очередь transcribe_gpu =====

@app.task(bind=True, name="tasks.download_audio", queue="transcribe_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")

@app.task(bind=True, name="tasks.preview_transcribe", queue="transcribe_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    # convert
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"), None)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(src, wav)
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
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

    # store & broadcast
    r.set(f"preview_result:{upload_id}", json.dumps(preview, ensure_ascii=False))
    state = {"status": "preview_done", "preview": preview}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    # callbacks
    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb, json={"event": "preview_complete", "external_id": upload_id}, timeout=5)
        except: pass

    # next
    transcribe_segments.delay(upload_id, correlation_id)

@app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    model = get_whisper_model()
    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segs, _ = model.transcribe(str(wav), word_timestamps=True, **opts)

    out = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(exist_ok=True, parents=True)
    (d / "transcript.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    state = {"status": "transcript_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb, json={"event": "transcript_complete","external_id": upload_id}, timeout=5)
        except: pass

    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

# Diarization оставляем в своей очереди
@app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    ann = get_clustering_diarizer().apply({"audio": str(wav)})

    segs = [{"start": float(seg.start), "end": float(seg.end), "speaker": spk}
            for seg, _, spk in ann.itertracks(yield_label=True)]
    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(exist_ok=True, parents=True)
    (d / "diarization.json").write_text(json.dumps(segs, indent=2, ensure_ascii=False), encoding="utf-8")

    state = {"status": "diarization_done", "preview": None}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    for cb in json.loads(r.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(cb, json={"event": "diarization_complete","external_id": upload_id}, timeout=5)
        except: pass

@app.task(name="tasks.cleanup_old_uploads")
def cleanup_old_uploads():
    cutoff = time.time() - settings.FILE_RETENTION_DAYS * 86400
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)