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

# --- Speaker stitching / embedding ---
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
    for attempt in range(1, 6):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=(5, 30))
            code = resp.status_code
            if 200 <= code < 300 or code == 405:
                logger.info(f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} succeeded ({code})")
                return
            if 400 <= code < 500:
                logger.error(f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} returned {code}, aborting")
                return
            logger.warning(f"[WEBHOOK] {event_type} returned {code}, retrying (attempt {attempt})")
        except requests.RequestException as e:
            logger.warning(f"[WEBHOOK] {event_type} network error (attempt {attempt}): {e}")
        time.sleep(30)
    logger.error(f"[WEBHOOK] {event_type} failed after 5 attempts")


def probe_audio(src: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)],
        capture_output=True, text=True
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


def prepare_preview_segment(upload_id: str) -> subprocess.Popen:
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    return subprocess.Popen([
        "ffmpeg", "-y", "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
        "-ss", "0", "-t", str(settings.PREVIEW_LENGTH_S),
        "-i", str(src),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        "-f", "wav", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


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
        buf.update(start=None, end=None, speaker=None, text=[])
    for seg in segments:
        txt = seg["text"].strip()
        if not txt:
            continue
        if buf["start"] is None:
            buf["start"], buf["speaker"] = seg["start"], seg.get("speaker")
        if buf["end"] is not None and (seg["start"] - buf["end"] > SILENCE_GAP_S):
            flush_buffer()
            buf["start"], buf["speaker"] = seg["start"], seg.get("speaker")
        buf["end"] = seg["end"]
        buf["text"].append(txt)
        if sentence_end_re.search(txt) or sum(len(t.split()) for t in buf["text"]) >= MAX_WORDS:
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
        try:
            _whisper_model = WhisperModel(path, device=device, compute_type=compute)
            logger.info(f"[WHISPER] loaded model from {path} on {device} with compute_type={compute}")
        except (RuntimeError, ValueError) as e:
            logger.warning(f"[WHISPER] GPU unavailable ({e}), falling back to CPU/INT8")
            settings.WHISPER_DEVICE = "cpu"
            _whisper_model = WhisperModel(path, device="cpu", compute_type="int8")
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
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            logger.warning(f"[STITCH] speechbrain not available, cannot do speaker stitching: {e}")
            raise
        savedir = Path(settings.DIARIZER_CACHE_DIR) / "spkrec-ecapa-voxceleb"
        _speaker_embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir)
        )
        logger.info("[STITCH] loaded speaker embedding model from speechbrain/spkrec-ecapa-voxceleb")
    return _speaker_embedding_model


def global_cluster_speakers(raw: List[Dict[str, Any]], wav: Path, upload_id: str) -> List[Dict[str, Any]]:
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
        start_sample = int(seg["start"] * sr)
        end_sample   = int(seg["end"]   * sr)
        wf = waveform[:, start_sample:end_sample]
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

# Экспорт для API
merge_speakers = global_cluster_speakers


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
        logger.info(f"[{upload_id}] preparing WAV")
        prepare_wav(upload_id)
        logger.info(f"[{upload_id}] WAV ready")
    except Exception as e:
        logger.error(f"[{upload_id}] convert_to_wav_and_preview failed: {e}", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        send_webhook_event("processing_failed", upload_id, {"stage": "wav", "error": str(e)})
        return

    preview_transcribe.delay(upload_id, correlation_id)


@app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] preview_transcribe received")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    # fallback to CPU if GPUs busy
    try:
        inspector = app.control.inspect()
        active = inspector.active() or {}
        heavy = sum(1 for tasks in active.values() for t in tasks
                    if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments"))
        if heavy >= 2:
            logger.info(f"[{upload_id}] GPUs busy ({heavy}), falling back to CPU for preview")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] failed to inspect workers for preview fallback")

    try:
        proc = prepare_preview_segment(upload_id)
        model = get_whisper_model()
        segments_gen, _ = model.transcribe(
            proc.stdout,
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
        )
        proc.stdout.close()
        proc.wait()

        segments = list(segments_gen)
        for seg in segments:
            r.publish(
                f"progress:{upload_id}",
                json.dumps({
                    "status": "preview_partial",
                    "fragment": {"start": seg.start, "end": seg.end, "text": seg.text}
                })
            )

        preview = {
            "text": "".join(s.text for s in segments),
            "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        }
        out = Path(settings.RESULTS_FOLDER) / upload_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "preview_transcript.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))

        r.publish(f"progress:{upload_id}", json.dumps({"status": "preview_done", "preview": preview}))
        send_webhook_event("preview_completed", upload_id, {"preview": preview})

        transcribe_segments.delay(upload_id, correlation_id)

    except Exception as e:
        logger.error(f"[{upload_id}] preview_transcribe failed: {e}", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        send_webhook_event("processing_failed", upload_id, {"stage": "preview", "error": str(e)})
        return


@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] transcribe_segments received")
    wav, duration = prepare_wav(upload_id)

    total_chunks = math.ceil(duration / settings.CHUNK_LENGTH_S) if duration > settings.CHUNK_LENGTH_S else 1
    if total_chunks > 1:
        logger.info(f"[{upload_id}] total transcription chunks: {total_chunks}")

    offset = 0.0
    idx = 0
    raw_segs: List[Any] = []

    def _transcribe_with_vad(source, offset=0.0):
        segs, _ = get_whisper_model().transcribe(
            source,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": int(settings.SENTENCE_MAX_GAP_S * 1000),
                "speech_pad_ms": 200,
            },
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
        )
        for s in segs:
            s.start += offset
            s.end += offset
        return list(segs)

    while offset < duration:
        length = min(settings.CHUNK_LENGTH_S, duration - offset)
        if total_chunks > 1:
            logger.info(f"[{upload_id}] transcription chunk {idx+1}/{total_chunks}: {offset:.1f}s→{offset+length:.1f}s")
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-threads", str(settings.FFMPEG_THREADS),
             "-ss", str(offset), "-t", str(length),
             "-i", str(wav), "-f", "wav", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        chunk_segs = _transcribe_with_vad(p.stdout, offset)
        p.stdout.close()
        p.wait()
        raw_segs.extend(chunk_segs)
        offset += length
        idx += 1

    flat = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segs]
    flat.sort(key=lambda x: x["start"])
    sentences = group_into_sentences(flat)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))

    Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True).publish(
        f"progress:{upload_id}", json.dumps({"status": "transcript_done"})
    )
    send_webhook_event("transcription_completed", upload_id, {"transcript": sentences})


@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] diarize_full received")
    wav, duration = prepare_wav(upload_id)
    raw_chunk_limit = int(getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0))
    using_chunking = bool(raw_chunk_limit and duration > raw_chunk_limit)

    total_chunks = math.ceil(duration / raw_chunk_limit) if using_chunking else 1
    logger.info(f"[{upload_id}] total diarization chunks: {total_chunks}")

    offset = 0.0
    idx = 0
    raw: List[Dict[str, Any]] = []
    pipeline = get_diarization_pipeline()

    while offset < duration:
        length = min(raw_chunk_limit, duration - offset)
        if using_chunking:
            logger.info(f"[{upload_id}] diarization chunk {idx+1}/{total_chunks}: {offset:.1f}s→{offset+length:.1f}s")
        tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{int(offset)}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
            "-ss", str(offset), "-t", str(length),
            "-i", str(wav), str(tmp)
        ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        ann = pipeline(str(tmp))
        for s, _, spk in ann.itertracks(yield_label=True):
            raw.append({
                "start": float(s.start) + offset,
                "end":   float(s.end)   + offset,
                "speaker": spk
            })
        tmp.unlink(missing_ok=True)
        offset += length
        idx += 1

    raw.sort(key=lambda x: x["start"])

    if SPEAKER_STITCH_ENABLED and using_chunking:
        raw = global_cluster_speakers(raw, wav, upload_id)
    else:
        logger.debug(f"[{upload_id}] skipping speaker stitching/cluster (using_chunking={using_chunking})")

    diar_sentences = []
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

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "diarization.json").write_text(json.dumps(diar_sentences, ensure_ascii=False, indent=2))
    send_webhook_event("diarization_completed", upload_id, {"diarization": diar_sentences})


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