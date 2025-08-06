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
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from torchaudio import load as load_wav

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
_speaker_embedding_model = None  # type: ignore

# --- Speaker stitching / embedding thresholds ---
SPEAKER_STITCH_ENABLED         = getattr(settings, "SPEAKER_STITCH_ENABLED", False)
SPEAKER_STITCH_THRESHOLD       = float(getattr(settings, "SPEAKER_STITCH_THRESHOLD", 0.75))
SPEAKER_STITCH_POOL_SIZE       = int(getattr(settings, "SPEAKER_STITCH_POOL_SIZE", 5))
SPEAKER_STITCH_EMA_ALPHA       = float(getattr(settings, "SPEAKER_STITCH_EMA_ALPHA", 0.4))
SPEAKER_STITCH_MERGE_THRESHOLD = float(getattr(settings, "SPEAKER_STITCH_MERGE_THRESHOLD", 0.75))

# --- Additional clustering controls ---
MIN_SEGMENT_LENGTH_S = float(getattr(settings, "MIN_SEGMENT_LENGTH_S", 0.2))
MIN_CLUSTER_SIZE     = int(getattr(settings, "MIN_CLUSTER_SIZE", 5))

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
            logger.warning(f"[WEBHOOK] {event_type} network error (attempt {attempt}/{max_attempts}) for {upload_id}: {e}")
        else:
            code = resp.status_code
            if 200 <= code < 300 or code == 405:
                logger.info(f"[WEBHOOK] {event_type} succeeded (attempt {attempt}) for {upload_id}")
                return
            if 400 <= code < 500:
                logger.error(f"[WEBHOOK] {event_type} returned {code} for {upload_id}, aborting")
                return
            logger.warning(f"[WEBHOOK] {event_type} returned {code} (attempt {attempt}), retrying")
        if attempt < max_attempts:
            time.sleep(30)
    logger.error(f"[WEBHOOK] {event_type} failed after {max_attempts} attempts for {upload_id}")

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
    wav, _ = prepare_wav(upload_id)
    return subprocess.Popen([
        "ffmpeg", "-y", "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
        "-ss", "0", "-t", str(settings.PREVIEW_LENGTH_S),
        "-i", str(wav),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def group_into_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    SILENCE_GAP_S = getattr(settings, "SENTENCE_MAX_GAP_S", 0.5)
    MAX_WORDS     = getattr(settings, "SENTENCE_MAX_WORDS", 50)
    sentences, buf = [], {"start": None, "end": None, "speaker": None, "text": []}
    sentence_end_re = re.compile(r"[\.!\?]$")
    def flush_buffer():
        if buf["text"] and buf["start"] is not None:
            sentences.append({
                "start": buf["start"], "end": buf["end"],
                "speaker": buf["speaker"], "text": " ".join(buf["text"])
            })
        buf.update({"start": None, "end": None, "speaker": None, "text": []})
    for seg in segments:
        txt = seg["text"].strip()
        if not txt:
            continue
        if buf["start"] is None:
            buf["start"], buf["speaker"] = seg["start"], seg.get("speaker")
        if buf["end"] is not None and seg["start"] - buf["end"] > SILENCE_GAP_S:
            flush_buffer()
            buf["start"], buf["speaker"] = seg["start"], seg.get("speaker")
        buf["end"] = seg["end"]
        buf["text"].append(txt)
        word_count = sum(len(t.split()) for t in buf["text"])
        if sentence_end_re.search(txt) or word_count >= MAX_WORDS:
            flush_buffer()
    flush_buffer()
    return sentences

def merge_speakers(
    transcript: List[Dict[str, Any]],
    diar: List[Dict[str, Any]],
    pad: float = 0.2,
) -> List[Dict[str, Any]]:
    if not diar:
        return [{**t, "speaker": None} for t in transcript]
    from bisect import bisect_left
    diar_sorted       = sorted(diar, key=lambda d: d["start"])
    transcript_sorted = sorted(transcript, key=lambda t: t["start"])
    starts            = [d["start"] for d in diar_sorted]
    def nearest(idx: int, t0: float, t1: float):
        if idx <= 0: return diar_sorted[0]
        if idx >= len(diar_sorted): return diar_sorted[-1]
        b, a = diar_sorted[idx-1], diar_sorted[idx]
        db = max(0.0, t0 - b["end"]); da = max(0.0, a["start"] - t1)
        return b if db <= da else a

    merged = []
    for t in transcript_sorted:
        t0, t1 = max(0.0, t["start"]-pad), t["end"]+pad
        i = bisect_left(starts, t1)
        cands = [d for d in diar_sorted[max(0,i-8):i+8]
                 if not (d["end"]<=t0 or d["start"]>=t1)]
        best = (max(cands, key=lambda d: max(0.0, min(d["end"],t1)-max(d["start"],t0)))
                if cands else nearest(i,t0,t1))
        merged.append({**t, "speaker": best["speaker"]})
    return merged

def get_whisper_model(model_override: str = None):
    global _whisper_model
    device  = settings.WHISPER_DEVICE.lower()
    compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "float16" if device.startswith("cuda") else "int8").lower()
    if model_override:
        return WhisperModel(model_override, device=device, compute_type=compute)
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        try:
            path = download_model(model_id, cache_dir=settings.HUGGINGFACE_CACHE_DIR,
                                  local_files_only=(device=="cpu"))
        except Exception:
            path = model_id
        if device=="cpu" and compute in ("fp16","float16"):
            compute = "int8"
        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info(f"[WHISPER] loaded model from {path} on {device}")
    return _whisper_model

def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        model_id = getattr(settings, "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1")
        _diarization_pipeline = PyannotePipeline.from_pretrained(
            model_id, use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.DIARIZER_CACHE_DIR
        )
        logger.info(f"[DIARIZE] loaded pipeline {model_id}")
    return _diarization_pipeline

def get_speaker_embedding_model():
    global _speaker_embedding_model
    if _speaker_embedding_model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = Path(settings.DIARIZER_CACHE_DIR) / "spkrec-ecapa-voxceleb"
        _speaker_embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(savedir)
        )
        logger.info("[EMBED] loaded speaker embedding model")
    return _speaker_embedding_model

def global_cluster_speakers(
    raw: List[Dict[str, Any]],
    wav: Path,
    upload_id: str
) -> List[Dict[str, Any]]:
    """
    Собираем эмбеддинги каждого сегмента и кластеризуем их по cosine distance.
    Фильтруем сегменты короче MIN_SEGMENT_LENGTH_S, кластеризуем остальные,
    затем короткие и мелкие кластеры перераспределяем к ближайшим по времени.
    """
    # разделим на короткие и валидные
    valid_indices = [i for i, seg in enumerate(raw) if seg["end"] - seg["start"] >= MIN_SEGMENT_LENGTH_S]
    short_indices = [i for i in range(len(raw)) if i not in valid_indices]

    waveform, sr = load_wav(str(wav))
    if sr != 16000:
        waveform = Resample(sr, 16000)(waveform)
        sr = 16000

    model = get_speaker_embedding_model()

    # формируем эмбеддинги только для валидных
    embeddings = []
    for i in valid_indices:
        seg = raw[i]
        s0, s1 = int(seg["start"] * sr), int(seg["end"] * sr)
        wf = waveform[:, s0:s1]
        if wf.numel() == 0:
            emb = torch.zeros(model._embedding_size if hasattr(model, "_embedding_size") else 192)
        else:
            with torch.no_grad():
                out = model.encode_batch(wf)
            emb_tensor = out.squeeze()
            emb = F.normalize(emb_tensor.flatten(), p=2, dim=0)
        embeddings.append(emb.cpu().numpy())

    if not embeddings:
        # нет валидных сегментов — возвращаем исходное
        return raw

    X = np.vstack(embeddings)
    thresh = 1.0 - SPEAKER_STITCH_MERGE_THRESHOLD
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=thresh
    ).fit(X)
    labels = clustering.labels_

    # посчитаем размеры кластеров и выделим мелкие
    from collections import Counter
    cnt = Counter(labels)
    small_clusters = {lab for lab, c in cnt.items() if c < MIN_CLUSTER_SIZE}

    # Если есть мелкие, переназначаем их в ближайший крупный по центроиду
    centroids = {}
    for lab in set(labels):
        inds = [j for j, l in enumerate(labels) if l == lab]
        centroids[lab] = np.mean(X[inds], axis=0)
        centroids[lab] /= np.linalg.norm(centroids[lab]) + 1e-8

    # перераспределяем мелкие кластеры
    for idx, lab in enumerate(labels):
        if lab in small_clusters:
            # найдем ближайший крупный кластер
            sims = {L: np.dot(centroids[lab], centroids[L]) for L in centroids if L not in small_clusters}
            best = max(sims, key=sims.get)
            labels[idx] = best

    # присваиваем лейблы обратно raw
    for pos, seg_idx in enumerate(valid_indices):
        raw[seg_idx]["speaker"] = f"spk_{labels[pos]}"

    # для коротких сегментов переназначаем к ближайшему по времени
    for i in short_indices:
        seg = raw[i]
        cand = [
            (abs(seg["start"] - raw[j]["start"]), raw[j]["speaker"])
            for j in valid_indices
        ]
        if cand:
            raw[i]["speaker"] = min(cand, key=lambda x: x[0])[1]

    return raw

@worker_process_init.connect
def preload_on_startup(**kwargs):
    if _HF_AVAILABLE:
        try: get_whisper_model()
        except: pass
    if _PN_AVAILABLE:
        try: get_diarization_pipeline()
        except: pass

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
        insp = app.control.inspect()
        active = insp.active() or {}
        heavy = sum(1 for tasks in active.values() for t in tasks
                    if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments"))
        if heavy >= 2:
            logger.info(f"[{upload_id}] GPUs busy ({heavy}), fallback to CPU preview")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] inspector failed, using GPU")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    proc = prepare_preview_segment(upload_id)
    model = get_whisper_model()
    segs, _ = model.transcribe(
        proc.stdout, word_timestamps=True,
        **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
    )
    proc.stdout.close(); proc.wait()

    fragments = []
    for s in segs:
        fragments.append({"start": s.start, "end": s.end, "text": s.text})
        r.publish(f"progress:{upload_id}", json.dumps({
            "status": "preview_partial",
            "fragment": fragments[-1],
        }))

    preview = {
        "text": "".join(f["text"] for f in fragments),
        "timestamps": fragments
    }
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "preview_transcript.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status": "preview_done", "preview": preview}))
    send_webhook_event("preview_completed", upload_id, {"preview": preview})

    transcribe_segments.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] transcribe_segments received")
    try:
        logger.info(f"[{upload_id}] GPU memory before: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except:
        pass

    try:
        insp = app.control.inspect()
        active = insp.active() or {}
        heavy = sum(1 for tasks in active.values() for t in tasks
                    if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments"))
        if heavy >= 2 and self.request.delivery_info.get("routing_key") != "transcribe_cpu":
            logger.info(f"[{upload_id}] GPUs busy ({heavy}), reschedule CPU")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] inspector fallback failed")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav, duration = prepare_wav(upload_id)
    if not _HF_AVAILABLE:
        logger.error(f"[{upload_id}] whisper unavailable, abort")
        send_webhook_event("processing_failed", upload_id, None)
        return

    model = get_whisper_model()
    raw_segs: List[Any] = []

    def _transcribe_with_vad(source, offset: float = 0.0, chunk_idx: Optional[int] = None, total_chunks: Optional[int] = None):
        segs_gen, _ = model.transcribe(
            source, word_timestamps=True, vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": int(settings.SENTENCE_MAX_GAP_S * 1000),
                "speech_pad_ms": 200,
            },
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        segs = list(segs_gen)
        if chunk_idx is not None and total_chunks is not None:
            r.publish(f"progress:{upload_id}", json.dumps({
                "status": "transcribe_chunk_done",
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "segments": len(segs),
            }))
        for s in segs:
            s.start += offset
            s.end += offset
        return segs

    if duration <= settings.VAD_MAX_LENGTH_S:
        logger.info(f"[{upload_id}] short audio ({duration:.1f}s)")
        raw_segs = _transcribe_with_vad(str(wav))
    else:
        total_chunks = math.ceil(duration / settings.CHUNK_LENGTH_S)
        offset = 0.0
        for idx in range(total_chunks):
            length = min(settings.CHUNK_LENGTH_S, duration - offset)
            logger.info(f"[{upload_id}] transcribe chunk {idx+1}/{total_chunks}: {offset:.1f}s→{offset+length:.1f}s")
            p = subprocess.Popen(
                ["ffmpeg", "-y", "-threads", str(settings.FFMPEG_THREADS),
                 "-ss", str(offset), "-t", str(length), "-i", str(wav), "-f", "wav", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            segs = _transcribe_with_vad(p.stdout, offset, idx+1, total_chunks)
            p.stdout.close(); p.wait()
            raw_segs.extend(segs)
            offset += length

    flat = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segs]
    flat.sort(key=lambda x: x["start"])
    sentences = group_into_sentences(flat)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
    logger.info(f"[{upload_id}] transcription done ({len(sentences)} sentences)")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))
    send_webhook_event("transcription_completed", upload_id, {"transcript": sentences})

    try:
        logger.info(f"[{upload_id}] GPU memory after: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except:
        pass

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] diarize_full started")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    send_webhook_event("diarization_started", upload_id, None)

    wav, duration = prepare_wav(upload_id)
    raw_chunk_limit = int(getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0))
    using_chunking  = bool(raw_chunk_limit and duration > raw_chunk_limit)
    pipeline        = get_diarization_pipeline()
    raw: List[Dict[str, Any]] = []

    pad = float(getattr(settings, "DIARIZATION_CHUNK_PADDING_S", raw_chunk_limit * 0.1))

    if using_chunking:
        total_chunks = math.ceil(duration / raw_chunk_limit)
        offset       = 0.0
        for idx in range(total_chunks):
            length    = min(raw_chunk_limit, duration - offset)
            start_pad = max(0.0, offset - pad)
            end_pad   = min(duration, offset + length + pad)
            proc_dur  = end_pad - start_pad

            logger.info(f"[{upload_id}] diarize chunk {idx+1}/{total_chunks}: "
                        f"{start_pad:.1f}s→{end_pad:.1f}s (core {offset:.1f}-{offset+length:.1f})")
            tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{idx}.wav"
            subprocess.run([
                "ffmpeg", "-y",
                "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
                "-ss", str(start_pad), "-t", str(proc_dur),
                "-i", str(wav), str(tmp)
            ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

            for attempt in range(2):
                try:
                    ann = pipeline(str(tmp))
                    before = len(raw)
                    for s, _, spk in ann.itertracks(yield_label=True):
                        t0 = float(s.start) + start_pad
                        t1 = float(s.end)   + start_pad
                        c0 = max(t0, offset)
                        c1 = min(t1, offset + length)
                        if c1 > c0:
                            raw.append({"start": c0, "end": c1, "speaker": spk})
                    added = len(raw) - before
                    logger.info(f"[{upload_id}] chunk {idx+1} added {added} segments")
                    r.publish(f"progress:{upload_id}", json.dumps({
                        "status": "diarize_chunk_done",
                        "chunk_index": idx+1,
                        "total_chunks": total_chunks,
                        "added_segments": added,
                    }))
                    break
                except Exception as e:
                    logger.warning(f"[{upload_id}] chunk {idx+1} failed (attempt {attempt+1}): {e}")
                    try: torch.cuda.empty_cache()
                    except: pass
                    time.sleep(5)
            tmp.unlink(missing_ok=True)
            offset += length

        raw = global_cluster_speakers(raw, wav, upload_id)

    else:
        ann = pipeline(str(wav))
        for s, _, spk in ann.itertracks(yield_label=True):
            raw.append({"start": float(s.start), "end": float(s.end), "speaker": spk})

    raw.sort(key=lambda x: x["start"])
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
    (out / "diarization.json").write_text(json.dumps(diar_sentences, ensure_ascii=False, indent=2))
    logger.info(f"[{upload_id}] diarization_done: {len(diar_sentences)} segments")
    r.publish(f"progress:{upload_id}", json.dumps({
        "status": "diarization_done", "segments": len(diar_sentences)
    }))
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
    headers = {"Content-Type": "application/json", "X-WebHook-Secret": secret}
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