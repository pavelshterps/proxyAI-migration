import json
import logging
import subprocess
import time
import re
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict

import numpy as np
import requests
from redis import Redis
from celery.signals import worker_process_init
from sklearn.cluster import AgglomerativeClustering

from celery_app import app  # импорт Celery instance
from config.settings import settings

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

# --- Model availability flags & holders ---
_HF_AVAILABLE = False
_PN_AVAILABLE = False
_whisper_model = None
_diarization_pipeline = None

# --- Speaker embedding ---
SPEAKER_STITCH_ENABLED = getattr(settings, "SPEAKER_STITCH_ENABLED", True)
SPEAKER_STITCH_THRESHOLD = float(getattr(settings, "SPEAKER_STITCH_THRESHOLD", 0.75))
SPEAKER_STITCH_POOL_SIZE = int(getattr(settings, "SPEAKER_STITCH_POOL_SIZE", 5))
SPEAKER_STITCH_EMA_ALPHA = float(getattr(settings, "SPEAKER_STITCH_EMA_ALPHA", 0.4))
SPEAKER_STITCH_MERGE_THRESHOLD = float(getattr(settings, "SPEAKER_STITCH_MERGE_THRESHOLD", 0.95))
_speaker_embedding_model = None  # type: ignore

try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info("[INIT] faster-whisper available")
except ImportError as e:
    logger.warning(f"[INIT] faster-whisper not available: {e}")

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _PN_AVAILABLE = True
    logger.info("[INIT] pyannote.audio available")
except ImportError as e:
    logger.warning(f"[INIT] pyannote.audio not available: {e}")


# ---------------------- Helpers ----------------------

def send_webhook_event(event_type: str, upload_id: str, data: Optional[Any]):
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
    headers = {"Content-Type": "application/json", "X-WebHook-Secret": secret}

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=(5, 30))
        except requests.RequestException as e:
            logger.warning(
                f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} network error "
                f"(attempt {attempt}/{max_attempts}) for {upload_id}: {e}"
            )
        else:
            code = resp.status_code
            if 200 <= code < 300 or code == 405:
                logger.info(
                    f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} "
                    f"{'treated as success' if code == 405 else 'succeeded'} "
                    f"(attempt {attempt}/{max_attempts}) for {upload_id}"
                )
                return
            if 400 <= code < 500:
                logger.error(
                    f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} returned {code} "
                    f"for {upload_id}, aborting"
                )
                return
            logger.warning(
                f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} returned {code} "
                f"(attempt {attempt}/{max_attempts}), retrying"
            )
        if attempt < max_attempts:
            time.sleep(30)

    logger.error(
        f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} failed after "
        f"{max_attempts} attempts for {upload_id}"
    )


def probe_audio(src: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)],
        capture_output=True,
        text=True
    )
    info = {"duration": 0.0}
    try:
        j = json.loads(res.stdout)
        info["duration"] = float(j["format"].get("duration", 0.0))
        for s in j.get("streams", []):
            if s.get("codec_type") == "audio":
                info.update({
                    "codec_name": s.get("codec_name"),
                    "sample_rate": int(s.get("sample_rate", 0)),
                    "channels": int(s.get("channels", 0)),
                })
                break
    except Exception:
        pass
    return info


def prepare_wav(upload_id: str) -> (Path, float):
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    info = probe_audio(src)
    duration = info["duration"]

    if (
        src.suffix.lower() == ".wav"
        and info.get("codec_name") == "pcm_s16le"
        and info.get("sample_rate") == 16000
        and info.get("channels") == 1
    ):
        if src != target:
            src.rename(target)
        return target, duration

    subprocess.run([
        "ffmpeg", "-y", "-threads", str(settings.FFMPEG_THREADS),
        "-i", str(src),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", str(target),
    ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return target, duration


def group_into_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    SILENCE_GAP_S = getattr(settings, "SENTENCE_MAX_GAP_S", 0.5)
    MAX_WORDS = getattr(settings, "SENTENCE_MAX_WORDS", 50)

    sentences = []
    buf = {"start": None, "end": None, "speaker": None, "text": []}
    sentence_end_re = re.compile(r"[\.!\?]$")

    def flush_buffer():
        if buf["text"] and buf["start"] is not None:
            sentences.append({
                "start": buf["start"],
                "end": buf["end"],
                "speaker": buf["speaker"],
                "text": " ".join(buf["text"]),
            })
        buf["start"] = buf["end"] = buf["speaker"] = None
        buf["text"] = []

    for seg in segments:
        txt = seg["text"].strip()
        if not txt:
            continue

        if buf["start"] is None:
            buf["start"] = seg["start"]
            buf["speaker"] = seg.get("speaker")

        if buf["end"] is not None and (seg["start"] - buf["end"] > SILENCE_GAP_S):
            flush_buffer()
            buf["start"] = seg["start"]
            buf["speaker"] = seg.get("speaker")

        buf["end"] = seg["end"]
        buf["text"].append(txt)

        word_count = sum(len(t.split()) for t in buf["text"])
        if sentence_end_re.search(txt) or word_count >= MAX_WORDS:
            flush_buffer()

    flush_buffer()
    return sentences


def get_whisper_model(model_override: str = None):
    global _whisper_model
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(
        settings,
        "WHISPER_COMPUTE_TYPE",
        "float16" if device.startswith("cuda") else "int8",
    ).lower()

    if model_override:
        logger.info(f"[WHISPER] loading override model {model_override}")
        return WhisperModel(model_override, device=device, compute_type=compute)

    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        try:
            path = download_model(
                model_id,
                cache_dir=settings.HUGGINGFACE_CACHE_DIR,
                local_files_only=(device == "cpu"),
            )
        except Exception:
            path = model_id

        if device == "cpu" and compute in ("fp16", "float16"):
            compute = "int8"

        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info(f"[WHISPER] loaded model from {path} on {device} with compute_type={compute}")

    return _whisper_model


def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        model_id = getattr(settings, "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1")
        _diarization_pipeline = PyannotePipeline.from_pretrained(
            model_id,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.DIARIZER_CACHE_DIR,
        )
        logger.info(f"[DIARIZE] loaded pipeline {model_id}")
    return _diarization_pipeline


def get_speaker_embedding_model():
    global _speaker_embedding_model
    if _speaker_embedding_model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = Path(settings.DIARIZER_CACHE_DIR) / "spkrec-ecapa-voxceleb"
        _speaker_embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir)
        )
        logger.info("[EMBED] loaded speaker embedding model")
    return _speaker_embedding_model


def global_cluster_speakers(
    raw: List[Dict[str, Any]],
    wav: Path,
    upload_id: str
) -> List[Dict[str, Any]]:
    """
    Собираем эмбеддинги для каждого сегмента и склеиваем
    их в кластеры по cosine-affinity.
    """
    import torch
    from torchaudio.transforms import Resample
    from torchaudio import load as load_wav

    waveform, sr = load_wav(str(wav))
    if sr != 16000:
        waveform = Resample(sr, 16000)(waveform)
        sr = 16000

    model = get_speaker_embedding_model()
    embeddings = []
    for seg in raw:
        s0, s1 = int(seg["start"] * sr), int(seg["end"] * sr)
        wf = waveform[:, s0:s1]
        if wf.numel() == 0:
            emb = torch.zeros(model.meta["embedding_size"])
        else:
            if wf.size(0) > 1:
                wf = wf.mean(dim=0, keepdim=True)
            with torch.no_grad():
                emb = model.encode_batch(wf).squeeze()
            emb = torch.nn.functional.normalize(emb.flatten(), p=2, dim=0)
        embeddings.append(emb.cpu().numpy())

    X = np.vstack(embeddings)
    # расстояние threshold = 1 - cosine_similarity_threshold
    thresh = 1 - SPEAKER_STITCH_MERGE_THRESHOLD
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="cosine",
        linkage="average",
        distance_threshold=thresh
    ).fit(X)

    logger.info(f"[{upload_id}] global clustering of {len(raw)} segments")
    for seg, lbl in zip(raw, clustering.labels_):
        seg["speaker"] = f"spk_{lbl}"
    return raw


@worker_process_init.connect
def preload_on_startup(**kwargs):
    if _HF_AVAILABLE:
        get_whisper_model()
    if _PN_AVAILABLE:
        get_diarization_pipeline()


# --- Celery tasks ---

@app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] convert_to_wav_and_preview received")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status": "processing_started"}))
    send_webhook_event("processing_started", upload_id, None)

    try:
        prepare_wav(upload_id)
        logger.info(f"[{upload_id}] WAV ready")
    except Exception as e:
        logger.error(f"[{upload_id}] WAV prep failed: {e}")
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        send_webhook_event("processing_failed", upload_id, None)
        return

    preview_transcribe.delay(upload_id, correlation_id)


@app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] preview_transcribe received")
    try:
        inspector = app.control.inspect()
        active = inspector.active() or {}
        heavy = sum(
            1
            for tasks in active.values()
            for t in tasks
            if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments")
        )
        if heavy >= 2:
            logger.info(f"[{upload_id}] GPUs busy ({heavy}), fallback to CPU")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] inspector failed, using GPU")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    proc = prepare_wav(upload_id)[0].parent / f"{upload_id}.wav"
    # ... (тут ваша логика мини-транскрипции preview)
    # после preview:
    send_webhook_event("preview_completed", upload_id, {"preview": {}})  # адаптируйте


@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    # ... (логика транскрипции полного текста, без изменений)
    pass


@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full started")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    send_webhook_event("diarization_started", upload_id, None)

    wav, duration = prepare_wav(upload_id)
    raw_chunk_limit = int(getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0) or 0)
    using_chunking = bool(raw_chunk_limit and duration > raw_chunk_limit)
    pipeline = get_diarization_pipeline()
    raw: List[Dict[str, Any]] = []

    if using_chunking:
        total_chunks = math.ceil(duration / raw_chunk_limit)
        offset = 0.0
        for chunk_idx in range(total_chunks):
            length = min(raw_chunk_limit, duration - offset)
            tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{chunk_idx}.wav"
            subprocess.run([
                "ffmpeg", "-y",
                "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
                "-ss", str(offset), "-t", str(length),
                "-i", str(wav), str(tmp)
            ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

            for attempt in range(2):
                try:
                    ann = pipeline(str(tmp))
                    before = len(raw)
                    for s, _, spk in ann.itertracks(yield_label=True):
                        raw.append({
                            "start": float(s.start) + offset,
                            "end":   float(s.end)   + offset,
                            "speaker": spk
                        })
                    logger.info(f"[{upload_id}] chunk {chunk_idx+1}/{total_chunks} → +{len(raw)-before} segs")
                    r.publish(f"progress:{upload_id}", json.dumps({
                        "status": "diarize_chunk_done",
                        "chunk_index": chunk_idx,
                        "added_segments": len(raw)-before,
                    }))
                    break
                except Exception as e:
                    logger.warning(f"[{upload_id}] chunk {chunk_idx} failed, retry {attempt+1}: {e}")
                    torch.cuda.empty_cache() if "torch" in globals() else None
                    time.sleep(5)
            tmp.unlink(missing_ok=True)
            offset += length

        # **Полный embedding-кластеринг сразу по всем сегментам**:
        raw = global_cluster_speakers(raw, wav, upload_id)

    else:
        ann = pipeline(str(wav))
        for s, _, spk in ann.itertracks(yield_label=True):
            raw.append({
                "start": float(s.start),
                "end":   float(s.end),
                "speaker": spk
            })

    raw.sort(key=lambda x: x["start"])

    # Сборка "диар-сентенсов"
    diar_sentences: List[Dict[str, Any]] = []
    buf: Optional[Dict[str, Any]] = None
    for seg in raw:
        if buf and buf["speaker"] == seg["speaker"] and seg["start"] - buf["end"] < 0.1:
            buf["end"] = seg["end"]
        else:
            if buf:
                diar_sentences.append(buf)
            buf = dict(seg)
    if buf:
        diar_sentences.append(buf)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "diarization.json").write_text(
        json.dumps(diar_sentences, ensure_ascii=False, indent=2)
    )
    logger.info(f"[{upload_id}] diarization_done: {len(diar_sentences)} segments")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarization_done", "segments": len(diar_sentences)}))
    send_webhook_event("diarization_completed", upload_id, {"diarization": diar_sentences})


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