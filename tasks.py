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
            logger.warning(f"Compute '{compute}' unsupported on CPU; falling back to int8")
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
            settings.VAD_MODEL_PATH,
            use_auth_token=settings.HUGGINGFACE_TOKEN
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
        try: get_vad().apply({"audio": str(sample)})
        except: pass
        try: get_clustering_diarizer().apply({"audio": str(sample)})
        except: pass


# === 1) Preview on CPU ===
@app.task(bind=True, name="tasks.preview_transcribe", queue="preview_cpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    upl = Path(settings.UPLOAD_FOLDER)
    cands = list(upl.glob(f"{upload_id}.*"))
    if not cands:
        logger.error(f"[{correlation_id}] no source for {upload_id}")
        return

    wav = upl / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(cands[0], wav)
    except Exception as e:
        logger.error(f"[{correlation_id}] conversion error: {e}")
        return

    model = get_whisper_model()
    opts  = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segs, _ = model.transcribe(str(wav_path), word_timestamps=True, **opts)

    preview = {"text": "", "timestamps": []}
    for s in segs:
        if s.start >= settings.PREVIEW_LENGTH_S:
            break
        preview["text"] += s.text
        preview["timestamps"].append({
            "start": s.start, "end": s.end, "text": s.text
        })

    total_chunks = max(1, int((segs[-1].end // settings.CHUNK_LENGTH_S) + 1))
    state = {
        "status": "preview_done",
        "preview": preview,
        "chunks_total": total_chunks,
        "chunks_done": 0,
        "diarize_requested": False
    }

    r.set(f"preview_result:{upload_id}", json.dumps(preview, ensure_ascii=False))
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    split_audio.delay(upload_id, correlation_id)


# === 2) Split into fixed‐length chunks ===
@app.task(bind=True, name="tasks.split_audio", queue="split_cpu")
def split_audio(self, upload_id: str, correlation_id: str):
    upl = Path(settings.UPLOAD_FOLDER)
    wav = upl / f"{upload_id}.wav"
    if not wav.exists(): return

    audio = AudioSegment.from_file(str(wav))
    chunk_ms = int(settings.CHUNK_LENGTH_S * 1000)
    paths = []
    for i in range(0, len(audio), chunk_ms):
        out = upl / f"{upload_id}_chunk_{i//chunk_ms}.wav"
        audio[i : i+chunk_ms].export(str(out), format="wav")
        paths.append(str(out))

    for idx, p in enumerate(paths):
        dispatch_transcription.delay(upload_id, idx, p, correlation_id)


# === 3) Dispatch to GPU ===
@app.task(bind=True, name="tasks.dispatch_transcription", queue="dispatch_cpu")
def dispatch_transcription(self, upload_id: str, idx: int, wav_path: str, correlation_id: str):
    app.send_task(
        "tasks.transcribe_chunk",
        args=(upload_id, idx, wav_path, correlation_id),
        queue="transcribe_gpu"
    )


# === 4) GPU: Transcribe one chunk ===
@app.task(bind=True, name="tasks.transcribe_chunk", queue="transcribe_gpu")
def transcribe_chunk(self, upload_id: str, idx: int, wav_path: str, correlation_id: str):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    model = get_whisper_model()
    opts  = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segs, _ = model.transcribe(wav_path, word_timestamps=True, **opts)

    offset = idx * settings.CHUNK_LENGTH_S
    out = []
    for s in segs:
        out.append({
            "start": s.start + offset,
            "end":   s.end   + offset,
            "text":  s.text
        })

    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"chunk_{idx}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    collect_transcription.delay(upload_id, correlation_id)


# === 5) CPU: Collect all chunks ===
@app.task(bind=True, name="tasks.collect_transcription", queue="collect_cpu")
def collect_transcription(self, upload_id: str, correlation_id: str):
    r   = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    d   = Path(settings.RESULTS_FOLDER) / upload_id
    js  = sorted(d.glob("chunk_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    st  = json.loads(r.get(f"progress:{upload_id}") or "{}")
    tot = st.get("chunks_total", 0)

    if len(js) < tot:
        st["chunks_done"] = len(js)
        r.set(f"progress:{upload_id}", json.dumps(st, ensure_ascii=False))
        r.publish(f"progress:{upload_id}", json.dumps(st, ensure_ascii=False))
        return

    merged = []
    for f in js:
        merged.extend(json.loads(f.read_text(encoding="utf-8")))
    (d / "transcript.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    st["status"]      = "transcript_done"
    st["chunks_done"] = tot
    r.set(f"progress:{upload_id}", json.dumps(st, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(st, ensure_ascii=False))

    if st.get("diarize_requested"):
        app.send_task(
            "tasks.diarize_full",
            args=(upload_id, correlation_id),
            queue="diarize_gpu"
        )


# === 6) GPU: Speaker diarization ===
@app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    r   = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav.exists(): return

    ann = get_clustering_diarizer().apply({"audio": str(wav)})
    segs = []
    for seg, _, spk in ann.itertracks(yield_label=True):
        segs.append({
            "start": float(seg.start),
            "end":   float(seg.end),
            "speaker": spk
        })

    d = Path(settings.RESULTS_FOLDER) / upload_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "diarization.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    st = json.loads(r.get(f"progress:{upload_id}") or "{}")
    st["status"]            = "diarization_done"
    st["diarize_requested"] = True
    r.set(f"progress:{upload_id}", json.dumps(st, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(st, ensure_ascii=False))


# === Cleanup ===
@app.task(name="tasks.cleanup_old_uploads")
def cleanup_old_uploads():
    cutoff = time.time() - settings.FILE_RETENTION_DAYS * 86400
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)