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

# --- утилита безопасной очистки CUDA-кеша ---
def _safe_empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# --- полировка через GPT ---
def polish_with_gpt(text: str) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "Ты редактор русского текста. Исправь ошибки и сделай текст читабельным."},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content

# --- Model availability flags & holders ---
_HF_AVAILABLE = False
_PN_AVAILABLE = False
_whisper_model = None
_diarization_pipeline = None  # текущий инстанс пайплайна с привязкой к устройству
_diarization_device = None    # "cuda" или "cpu" для созданного пайплайна

# --- Speaker stitching / embedding ---
SPEAKER_STITCH_ENABLED = bool(getattr(settings, "SPEAKER_STITCH_ENABLED", False))
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

def merge_speakers(
    transcript: List[Dict[str, Any]],
    diar: List[Dict[str, Any]],
    pad: float = 0.2,
) -> List[Dict[str, Any]]:
    if not diar:
        return [{**t, "speaker": None} for t in transcript]

    diar = sorted(diar, key=lambda d: d["start"])
    transcript = sorted(transcript, key=lambda t: t["start"])
    starts = [d["start"] for d in diar]

    from bisect import bisect_left

    def nearest(idx: int, t0: float, t1: float):
        if idx <= 0:
            return diar[0]
        if idx >= len(diar):
            return diar[-1]
        b, a = diar[idx - 1], diar[idx]
        db = max(0.0, t0 - b["end"])
        da = max(0.0, a["start"] - t1)
        return b if db <= da else a

    out = []
    for t in transcript:
        t0 = max(0.0, t["start"] - pad)
        t1 = t["end"] + pad
        i = bisect_left(starts, t1)
        cands = [
            d for d in diar[max(0, i - 8): i + 8]
            if not (d["end"] <= t0 or d["start"] >= t1)
        ]
        best = max(cands, key=lambda d: max(0.0, min(d["end"], t1) - max(d["start"], t0))) \
            if cands else nearest(i, t0, t1)
        out.append({**t, "speaker": best["speaker"]})
    return out

# ---------- Whisper ----------

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

# ---------- PyAnnote diarization pipeline ----------

def _build_diarization_pipeline(prefer_device: str) -> Any:
    """
    Создаёт пайплайн и переводит на устройство (cuda|cpu).
    Если не получилось, пробует CPU.
    """
    global _diarization_pipeline, _diarization_device
    model_id = getattr(settings, "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1")
    tok = getattr(settings, "HUGGINGFACE_TOKEN", None)

    pipeline = PyannotePipeline.from_pretrained(
        model_id,
        use_auth_token=tok,
        cache_dir=settings.DIARIZER_CACHE_DIR,
    )
    # pyannote 3.x поддерживает .to("cuda"/"cpu")
    try:
        pipeline.to(prefer_device)
        _diarization_pipeline = pipeline
        _diarization_device = prefer_device
        logger.info(f"[DIARIZE] loaded pipeline {model_id} on {prefer_device}")
        return _diarization_pipeline
    except Exception as e:
        logger.warning(f"[DIARIZE] move to {prefer_device} failed, fallback to CPU: {e}")
        try:
            pipeline.to("cpu")
            _diarization_pipeline = pipeline
            _diarization_device = "cpu"
            logger.info(f"[DIARIZE] loaded pipeline {model_id} on cpu")
            return _diarization_pipeline
        except Exception as e2:
            logger.error(f"[DIARIZE] CPU fallback failed: {e2}")
            raise

def get_diarization_pipeline(prefer_device: Optional[str] = None):
    """
    Возвращает существующий пайплайн, если он уже на нужном устройстве.
    Иначе пересобирает.
    """
    global _diarization_pipeline, _diarization_device
    if not _PN_AVAILABLE:
        raise RuntimeError("pyannote.audio not available")
    target = (prefer_device or ("cuda" if _has_cuda() else "cpu")).lower()
    if _diarization_pipeline is None or _diarization_device != target:
        _diarization_pipeline = _build_diarization_pipeline(target)
    return _diarization_pipeline

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# ---------- Optional SpeechBrain stitching ----------

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
        logger.info("[STITCH] loaded speaker embedding model")
    return _speaker_embedding_model

def stitch_speakers(raw: List[Dict[str, Any]], wav: Path, upload_id: str) -> List[Dict[str, Any]]:
    if not SPEAKER_STITCH_ENABLED:
        return raw

    unique_orig = set(seg.get("speaker") for seg in raw)
    if len(unique_orig) <= 1:
        logger.debug(f"[{upload_id}] only one original speaker {unique_orig}, skipping stitching")
        return raw

    try:
        import torch
        import torchaudio
        import torch.nn.functional as F

        model = get_speaker_embedding_model()

        waveform, sr = torchaudio.load(str(wav))
        if sr != 16000:
            from torchaudio.transforms import Resample
            resampler = Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        stitch_centroids: Dict[str, torch.Tensor] = {}
        stitch_histories: Dict[str, List[torch.Tensor]] = {}
        next_label_idx = 0

        def new_canonical_label():
            nonlocal next_label_idx
            label = f"spk_{next_label_idx}"
            next_label_idx += 1
            return label

        stitched: List[Dict[str, Any]] = []
        raw_sorted = sorted(raw, key=lambda x: x["start"])
        for seg in raw_sorted:
            start, end = seg["start"], seg["end"]
            if end <= start:
                stitched.append(seg)
                continue
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            if end_sample <= start_sample or start_sample >= waveform.size(1):
                stitched.append(seg)
                continue
            segment_waveform = waveform[:, start_sample:end_sample]
            if segment_waveform.numel() == 0:
                stitched.append(seg)
                continue
            if segment_waveform.size(0) > 1:
                segment_waveform = segment_waveform.mean(dim=0, keepdim=True)
            with torch.no_grad():
                emb = model.encode_batch(segment_waveform)
            emb = emb.squeeze()
            if emb.ndim > 1:
                emb = emb.flatten()
            emb = F.normalize(emb, p=2, dim=0)

            assigned_label = None
            best_sim = -1.0
            for canon_label, centroid in stitch_centroids.items():
                sim = torch.dot(emb, centroid).item()
                if sim > best_sim:
                    best_sim = sim
                    assigned_label = canon_label

            if assigned_label is not None and best_sim >= SPEAKER_STITCH_THRESHOLD:
                old_centroid = stitch_centroids[assigned_label]
                updated_centroid = F.normalize(
                    SPEAKER_STITCH_EMA_ALPHA * emb + (1 - SPEAKER_STITCH_EMA_ALPHA) * old_centroid, p=2, dim=0
                )
                stitch_centroids[assigned_label] = updated_centroid
                hist = stitch_histories[assigned_label]
                hist.append(emb)
                if len(hist) > SPEAKER_STITCH_POOL_SIZE:
                    hist.pop(0)
                logger.debug(f"[{upload_id}] reused speaker {assigned_label} (sim={best_sim:.3f})")
            else:
                assigned_label = new_canonical_label()
                stitch_centroids[assigned_label] = emb
                stitch_histories[assigned_label] = [emb]
                logger.debug(f"[{upload_id}] created new speaker label {assigned_label}")

            seg["speaker"] = assigned_label
            stitched.append(seg)

        # финальная агрегация похожих центроидов
        label_centroids: Dict[str, torch.Tensor] = {}
        for label, hist in stitch_histories.items():
            centroid = torch.stack(hist).mean(dim=0)
            centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
            label_centroids[label] = centroid

        adj: Dict[str, set] = {label: set() for label in label_centroids}
        labels = list(label_centroids.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                sim = torch.dot(label_centroids[a], label_centroids[b]).item()
                if sim >= SPEAKER_STITCH_MERGE_THRESHOLD:
                    adj[a].add(b)
                    adj[b].add(a)

        visited, merge_map = set(), {}
        for label in adj:
            if label in visited:
                continue
            stack, component = [label], []
            while stack:
                l = stack.pop()
                if l in visited:
                    continue
                visited.add(l)
                component.append(l)
                stack.extend(adj[l] - visited)
            if len(component) > 1:
                rep = sorted(component)[0]
                for l in component:
                    merge_map[l] = rep

        if merge_map:
            for seg in stitched:
                old = seg["speaker"]
                if old in merge_map:
                    new = merge_map[old]
                    if new != old:
                        logger.debug(f"[{upload_id}] merged speaker {old} -> {new}")
                        seg["speaker"] = new

        return stitched
    except Exception as e:
        logger.warning(f"[{upload_id}] speaker stitching failed, falling back to original labels: {e}")
        return raw

@worker_process_init.connect
def preload_on_startup(**kwargs):
    # Не поднимаем heavy-модели в воркере, который обслуживает только вебхуки
    queues = os.environ.get("CELERY_QUEUES", "")
    only_webhooks = False
    if queues:
        qset = {q.strip() for q in queues.split(",") if q.strip()}
        only_webhooks = ("webhooks" in qset) and all(q == "webhooks" for q in qset)

    if only_webhooks:
        logger.info("[PRELOAD] skip model preload in webhooks worker")
        return

    if _HF_AVAILABLE:
        try:
            get_whisper_model()
        except Exception as e:
            # fallback на CPU
            try:
                logger.warning(f"[PRELOAD] whisper GPU init failed, fallback to CPU: {e}")
                dev_bak = getattr(settings, "WHISPER_DEVICE", "cpu")
                setattr(settings, "WHISPER_DEVICE", "cpu")
                setattr(settings, "WHISPER_COMPUTE_TYPE", "int8")
                get_whisper_model()
                setattr(settings, "WHISPER_DEVICE", dev_bak)
            except Exception as e2:
                logger.error(f"[PRELOAD] whisper CPU fallback failed: {e2}")

    if _PN_AVAILABLE:
        try:
            get_diarization_pipeline(prefer_device="cuda" if _has_cuda() else "cpu")
        except Exception as e:
            logger.error(f"[PRELOAD] diarization pipeline init failed: {e}")

# --- Celery tasks ---

@app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] convert_to_wav_and_preview received")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status": "processing_started"}))
    deliver_webhook.delay("processing_started", upload_id, None)

    try:
        logger.info(f"[{upload_id}] preparing WAV")
        prepare_wav(upload_id)
        logger.info(f"[{upload_id}] WAV ready")
    except Exception as e:
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    preview_transcribe.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] preview_transcribe received")
    try:
        inspector = app.control.inspect()
        active = inspector.active() or {}
        heavy = 0
        for node_tasks in active.values():
            for t in node_tasks:
                if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments"):
                    heavy += 1
        if heavy >= 2:
            logger.info(f"[{upload_id}] GPUs busy ({heavy}), fallback to CPU for preview")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] failed to inspect workers, proceeding on GPU")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    proc = prepare_preview_segment(upload_id)
    model = get_whisper_model()
    segments_gen, _ = model.transcribe(
        proc.stdout,
        word_timestamps=True,
        **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
    )
    proc.stdout.close(); proc.wait()
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
    deliver_webhook.delay("preview_completed", upload_id, {"preview": preview})
    transcribe_segments.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] transcribe_segments received")
    try:
        import torch
        logger.info(f"[{upload_id}] GPU memory before: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

    try:
        inspector = app.control.inspect()
        active = inspector.active() or {}
        heavy = 0
        for node_tasks in active.values():
            for t in node_tasks:
                if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments") and t["name"] != "tasks.transcribe_segments":
                    heavy += 1
        if heavy >= 2 and self.request.delivery_info.get("routing_key") != "transcribe_cpu":
            logger.info(f"[{upload_id}] GPUs appear busy ({heavy}), rescheduling to CPU")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] failed to inspect workers for fallback logic")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav, duration = prepare_wav(upload_id)
    if not _HF_AVAILABLE:
        logger.error(f"[{upload_id}] whisper model unavailable, failing")
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    model = get_whisper_model()
    raw_segs: List[Any] = []

    def _transcribe_with_vad(source, offset: float = 0.0):
        segs, _ = model.transcribe(
            source,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": int(settings.SENTENCE_MAX_GAP_S * 1000),
                "speech_pad_ms": 200,
            },
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
        )
        result = []
        for s in segs:
            s.start += offset
            s.end += offset
            result.append(s)
        return result

    if duration <= settings.VAD_MAX_LENGTH_S:
        logger.info(f"[{upload_id}] short audio ({duration:.1f}s) → single VAD pass")
        raw_segs = _transcribe_with_vad(str(wav))
    else:
        total_chunks = math.ceil(duration / settings.CHUNK_LENGTH_S)
        processed_key = f"transcribe:processed_chunks:{upload_id}"
        processed = {int(x) for x in r.smembers(processed_key)}

        offset = 0.0
        chunk_idx = 0
        while offset < duration:
            if chunk_idx in processed:
                logger.info(f"[{upload_id}] skip chunk {chunk_idx+1}/{total_chunks} (already done)")
                offset += settings.CHUNK_LENGTH_S
                chunk_idx += 1
                continue

            length = min(settings.CHUNK_LENGTH_S, duration - offset)
            logger.info(f"[{upload_id}] transcribe chunk {chunk_idx+1}/{total_chunks}: {offset:.1f}s→{offset+length:.1f}s")
            try:
                p = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-threads", str(settings.FFMPEG_THREADS),
                        "-ss", str(offset), "-t", str(length),
                        "-i", str(wav), "-f", "wav", "pipe:1"
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                chunk_segs = _transcribe_with_vad(p.stdout, offset)
                p.stdout.close(); p.wait()

                raw_segs.extend(chunk_segs)
                r.sadd(processed_key, chunk_idx)
            except Exception as e:
                logger.error(f"[{upload_id}] error in transcribe chunk {chunk_idx+1}/{total_chunks}: {e}", exc_info=True)
                _safe_empty_cuda_cache()
            finally:
                offset += length
                chunk_idx += 1

        r.delete(processed_key)

    flat = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segs]
    flat.sort(key=lambda x: x["start"])
    sentences = group_into_sentences(flat)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript_original.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
    logger.info(f"[{upload_id}] saved original transcript ({len(sentences)} sentences)")

    try:
        raw_text = " ".join(s["text"] for s in sentences)
        polished = polish_with_gpt(raw_text)
        (out / "transcript.json").write_text(json.dumps({"text": polished}, ensure_ascii=False, indent=2))
        logger.info(f"[{upload_id}] saved polished transcript")
    except Exception as e:
        logger.warning(f"[{upload_id}] polishing failed: {e}")
        (out / "transcript.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))

    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))
    deliver_webhook.delay("transcription_completed", upload_id, {"transcript": sentences})

    try:
        import torch
        logger.info(f"[{upload_id}] GPU memory after: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

# ---------------------- SIMPLIFIED & ROBUST DIARIZATION ----------------------

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    """
    Упрощённая, но устойчивая диаризация:
    - один Celery-таск без вложенных сабтасков;
    - последовательная обработка чанков в том же процессе;
    - GPU→CPU fallback;
    - на любом исходе пишем diarization.json и шлём прогресс, чтобы фронт не зависал.
    """
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full started")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    deliver_webhook.delay("diarization_started", upload_id, None)

    diar_sentences: List[Dict[str, Any]] = []
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            import torch
            logger.info(f"[{upload_id}] GPU memory before diarization: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
        except ImportError:
            pass

        wav, duration = prepare_wav(upload_id)

        raw_chunk_limit = getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0)
        try:
            chunk_limit = int(raw_chunk_limit)
        except Exception:
            logger.warning(f"[{upload_id}] invalid DIARIZATION_CHUNK_LENGTH_S={raw_chunk_limit!r}, fallback 0")
            chunk_limit = 0

        using_chunking = bool(chunk_limit and duration > chunk_limit)
        logger.info(f"[{upload_id}] WAV ready for diarization: duration={duration:.1f}s; chunk_limit={chunk_limit}; chunking={using_chunking}")

        if not _PN_AVAILABLE:
            raise RuntimeError("pyannote.audio not available")

        # Создаём пайплайн (cuda если есть, иначе cpu). Если потом упадём — пересоздадим на cpu.
        try:
            pipeline = get_diarization_pipeline(prefer_device="cuda" if _has_cuda() else "cpu")
        except Exception as e:
            logger.warning(f"[{upload_id}] pipeline init failed on GPU, retry CPU: {e}")
            pipeline = get_diarization_pipeline(prefer_device="cpu")

        raw: List[Dict[str, Any]] = []

        def _run_chunk(offset: float, length: float):
            nonlocal pipeline
            tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{int(offset)}.wav"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
                        "-ss", str(offset), "-t", str(length),
                        "-i", str(wav), str(tmp)
                    ],
                    check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
                )
            except Exception as e:
                logger.error(f"[{upload_id}] ffmpeg chunk {offset:.1f}-{offset+length:.1f} failed: {e}")
                return 0

            try:
                ann = pipeline(str(tmp))
            except Exception as e:
                # Первый фейл: пробуем CPU fallback если были на GPU
                logger.warning(f"[{upload_id}] diarization chunk failed on {getattr(pipeline, 'device', 'unknown')}: {e}")
                try:
                    pipeline = get_diarization_pipeline(prefer_device="cpu")
                    ann = pipeline(str(tmp))
                    logger.info(f"[{upload_id}] chunk recovered on CPU")
                except Exception as e2:
                    logger.error(f"[{upload_id}] chunk failed on CPU too: {e2}")
                    return 0
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

            added = 0
            for s, _, spk in ann.itertracks(yield_label=True):
                raw.append({"start": float(s.start) + offset, "end": float(s.end) + offset, "speaker": spk})
                added += 1
            _safe_empty_cuda_cache()
            return added

        if using_chunking:
            total_chunks = math.ceil(duration / chunk_limit)
            offset = 0.0
            chunk_idx = 0
            while offset < duration:
                this_len = min(chunk_limit, duration - offset)
                logger.info(f"[{upload_id}] diarize chunk {chunk_idx+1}/{total_chunks}: {offset:.1f}s→{offset+this_len:.1f}s")
                _ = _run_chunk(offset, this_len)
                offset += this_len
                chunk_idx += 1
        else:
            # один проход без резки
            try:
                ann = pipeline(str(wav))
            except Exception as e:
                logger.warning(f"[{upload_id}] single-pass failed on GPU? retry CPU: {e}")
                pipeline = get_diarization_pipeline(prefer_device="cpu")
                ann = pipeline(str(wav))
            for s, _, spk in ann.itertracks(yield_label=True):
                raw.append({"start": float(s.start), "end": float(s.end), "speaker": spk})

        raw.sort(key=lambda x: x["start"])

        # Сшивка спикеров по желанию, но только если была резка (там нужнее всего)
        try:
            if SPEAKER_STITCH_ENABLED and using_chunking:
                raw = stitch_speakers(raw, wav, upload_id)
            else:
                logger.debug(f"[{upload_id}] skipping speaker stitching (enabled={SPEAKER_STITCH_ENABLED}, chunking={using_chunking})")
        except Exception as e:
            logger.warning(f"[{upload_id}] speaker stitching skipped due to error: {e}")

        # Слить соседние одноимённые сегменты
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
        try:
            (out_dir / "diarization.json").write_text(json.dumps(diar_sentences, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"[{upload_id}] failed to write diarization.json: {e}")
        try:
            import torch
            logger.info(f"[{upload_id}] GPU memory after diarization: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
        except ImportError:
            pass

        # В любом случае сигналим фронту, чтобы он не висел
        r.publish(
            f"progress:{upload_id}",
            json.dumps({"status": "diarization_done", "segments": len(diar_sentences)})
        )
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