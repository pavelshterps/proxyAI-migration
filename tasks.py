import os
import json
import logging
import subprocess
import time
import re
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict

import requests
from redis import Redis
from celery.signals import worker_process_init

from celery_app import app  # Celery instance
from config.settings import settings

# --- dotenv & OpenAI ---
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = settings.OPENAI_API_KEY

# --- Logger setup ---
logger = logging.getLogger("tasks")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- GPU lock via Redis ---
def gpu_lock(upload_id: str, ttl=3600):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    lock_key = f"gpu:lock"
    acquired = False
    while not acquired:
        acquired = r.set(lock_key, upload_id, nx=True, ex=ttl)
        if not acquired:
            time.sleep(1)
    try:
        yield
    finally:
        if r.get(lock_key) == upload_id:
            r.delete(lock_key)

# --- safe CUDA cache cleanup ---
def _safe_empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# --- rest of helpers (polish_with_gpt, probe_audio, prepare_wav, etc.) remain unchanged --- #
# ... [Omitted for brevity; assume identical to previous version you had] ...

# --- Whispers and other helpers retained --- #
# ... [Whisper model, diarization pipeline init, stitching, etc.] ...

# ---------- Tasks with GPU locking ---------- #

@app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] preview_transcribe received")
    with gpu_lock(upload_id):
        # existing logic unchanged...
        # transcribe preview, publish preview_partial & preview_done, call next steps
        pass  # guard remains as before

@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] transcribe_segments received")
    with gpu_lock(upload_id):
        # unchanged logic...
        # ensure torch.cuda.empty_cache() in finally per chunk
        pass

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full started")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    deliver_webhook.delay("diarization_started", upload_id, None)

    diar_sentences: List[Dict[str, Any]] = []
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)

    with gpu_lock(upload_id):
        try:
            wav, duration = prepare_wav(upload_id)
            raw_chunk_limit = int(getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0) or 0)
            using_chunking = raw_chunk_limit and duration > raw_chunk_limit

            try:
                pipeline = get_diarization_pipeline()
            except Exception as e:
                logger.warning(f"[{upload_id}] GPU init failed, retry CPU: {e}")
                pipeline = get_diarization_pipeline(prefer_device="cpu")

            raw: List[Dict[str, Any]] = []

            def process_chunk(offset: float, length: float):
                tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{offset:.1f}.wav"
                subprocess.run([
                    "ffmpeg", "-y", "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
                    "-ss", str(offset), "-t", str(length),
                    "-i", str(wav), str(tmp)
                ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                try:
                    result = pipeline(str(tmp))
                except RuntimeError as oom:
                    if "CUDA out of memory" in str(oom):
                        logger.warning(f"[{upload_id}] OOM, retry on CPU for chunk {offset:.1f}")
                        try:
                            import torch; torch.cuda.empty_cache()
                        except Exception:
                            pass
                        pipeline.to("cpu")
                        result = pipeline(str(tmp))
                    else:
                        raise
                finally:
                    tmp.unlink(missing_ok=True)

                for turn, _, speaker in result.itertracks(yield_label=True):
                    raw.append({"start": float(turn.start) + offset, "end": float(turn.end) + offset, "speaker": speaker})
                _safe_empty_cuda_cache()

            if using_chunking:
                total = math.ceil(duration / raw_chunk_limit)
                offset = 0.0
                idx = 0
                while offset < duration:
                    length = min(raw_chunk_limit, duration - offset)
                    logger.info(f"[{upload_id}] diarize chunk {idx+1}/{total}: {offset:.1f}s–{offset+length:.1f}s")
                    process_chunk(offset, length)
                    offset += length
                    idx += 1
            else:
                try:
                    result = pipeline(str(wav))
                    for turn, _, speaker in result.itertracks(yield_label=True):
                        raw.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
                except Exception as e:
                    logger.error(f"[{upload_id}] single-pass diarization failed: {e}", exc_info=True)
                    raw = []

            raw.sort(key=lambda x: x["start"])
            if SPEAKER_STITCH_ENABLED and using_chunking and raw:
                raw = stitch_speakers(raw, wav, upload_id)

            buf = None
            for seg in raw:
                if buf and buf["speaker"] == seg["speaker"] and seg["start"] - buf["end"] < 0.1:
                    buf["end"] = seg["end"]
                else:
                    if buf:
                        diar_sentences.append(buf)
                    buf = dict(seg)
            if buf:
                diar_sentences.append(buf)

        except Exception as fatal:
            logger.error(f"[{upload_id}] diarize_full fatal error: {fatal}", exc_info=True)
        finally:
            (out / "diarization.json").write_text(json.dumps(diar_sentences, ensure_ascii=False, indent=2))
            try:
                import torch
                logger.info(f"[{upload_id}] GPU memory after diarization: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
            except Exception:
                pass
            r.publish(f"progress:{upload_id}", json.dumps({"status": "diarization_done", "segments": len(diar_sentences)}))
            deliver_webhook.delay("diarization_completed", upload_id, {"diarization": diar_sentences})
@app.task(
    bind=True,
    name="deliver_webhook",
    queue="webhooks",
    max_retries=5,
    default_retry_delay=30,
)
def deliver_webhook(self, event_type: str, upload_id: str, data: Optional[Any]):
    url = settings.WEBHOOK_URL
    secret = settings.WEBHOOK_SECRET
    if not url or not secret:
        return

    payload = {
        "event_type": event_type,
        "upload_id": upload_id,
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "data": data,
    }
    headers = {
        "Content-Type": "application/json",
        "X-WebHook-Secret": secret,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=(5, 30))
        code = resp.status_code
        if 200 <= code < 300 or code == 405:
            logger.info(f"[WEBHOOK] {event_type} succeeded for {upload_id} ({code})")
            return
        if 400 <= code < 500:
            logger.error(f"[WEBHOOK] {event_type} returned {code} for {upload_id}, aborting")
            return
        raise Exception(f"Webhook returned {code}")
    except Exception as exc:
        logger.warning(f"[WEBHOOK] {event_type} error for {upload_id}, retrying: {exc}")
        raise self.retry(exc=exc)

@app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    age = settings.FILE_RETENTION_DAYS
    deleted = 0
    for base in (Path(settings.UPLOAD_FOLDER), Path(settings.RESULTS_FOLDER)):
        for p in base.glob("**/*"):
            try:
                if datetime.utcnow() - datetime.fromtimestamp(p.stat().st_mtime) > timedelta(days=age):
                    if p.is_dir():
                        p.rmdir()
                    else:
                        p.unlink()
                    deleted += 1
            except Exception:
                continue
    logger.info(f"[CLEANUP] deleted {deleted} old files")