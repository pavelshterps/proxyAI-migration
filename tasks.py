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

from celery_app import app  # импорт Celery instance
from config.settings import settings

# --- dotenv & OpenAI ---
from dotenv import load_dotenv
import openai
from openai.error import OpenAIError

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

# --- полировка через GPT ---
def polish_with_gpt(text: str) -> str:
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Ты редактор русского текста. Исправь ошибки и сделай текст читабельным."},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        logger.warning(f"[polish_with_gpt] OpenAI error: {e}")
        return text  # fallback to raw text

# --- Model availability flags & holders ---
_HF_AVAILABLE = False
_PN_AVAILABLE = False
_whisper_model = None
_diarization_pipeline = None

# --- Speaker stitching / embedding ---
SPEAKER_STITCH_ENABLED = getattr(settings, "SPEAKER_STITCH_ENABLED", False)
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
                logger.info(f"[WEBHOOK] {event_type} succeeded for {upload_id} ({code})")
                return
            if 400 <= code < 500:
                logger.error(f"[WEBHOOK] {event_type} returned {code} for {upload_id}, aborting")
                return
            logger.warning(f"[WEBHOOK] {event_type} returned {code}, retrying ({attempt}/5)")
        except requests.RequestException as e:
            logger.warning(f"[WEBHOOK] network error on {attempt}/5 for {upload_id}: {e}")
        time.sleep(30)
    logger.error(f"[WEBHOOK] {event_type} failed after 5 attempts for {upload_id}")

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
    # transcript: list of {start, end, text, speaker?}
    if not isinstance(transcript, list):
        return transcript  # fallback unmodified

    if not diar:
        return transcript

    diar_sorted = sorted(diar, key=lambda d: d["start"])
    transcript_sorted = sorted(transcript, key=lambda t: t["start"])
    starts = [d["start"] for d in diar_sorted]

    from bisect import bisect_left

    def nearest(idx: int, t0: float, t1: float):
        if idx <= 0:
            return diar_sorted[0]
        if idx >= len(diar_sorted):
            return diar_sorted[-1]
        before, after = diar_sorted[idx - 1], diar_sorted[idx]
        db = max(0.0, t0 - before["end"])
        da = max(0.0, after["start"] - t1)
        return before if db <= da else after

    out = []
    for t in transcript_sorted:
        t0 = max(0.0, t["start"] - pad)
        t1 = t["end"] + pad
        i = bisect_left(starts, t1)
        cands = [
            d for d in diar_sorted[max(0, i - 8): i + 8]
            if not (d["end"] <= t0 or d["start"] >= t1)
        ]
        best = max(cands, key=lambda d: max(0.0, min(d["end"], t1) - max(d["start"], t0))) \
            if cands else nearest(i, t0, t1)
        out.append({**t, "speaker": best["speaker"]})
    return out

def get_whisper_model(model_override: str = None):
    global _whisper_model
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(
        settings,
        "WHISPER_COMPUTE_TYPE",
        "float16" if device.startswith("cuda") else "int8",
    ).lower()

    if model_override:
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
        logger.info(f"[WHISPER] loaded model from {path} on {device}")
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
        logger.info("[STITCH] loaded speaker embedding model")
    return _speaker_embedding_model

def stitch_speakers(raw: List[Dict[str, Any]], wav: Path, upload_id: str) -> List[Dict[str, Any]]:
    if not SPEAKER_STITCH_ENABLED or len({seg.get("speaker") for seg in raw}) <= 1:
        return raw

    try:
        import torch, torchaudio, torch.nn.functional as F
        model = get_speaker_embedding_model()
        waveform, sr = torchaudio.load(str(wav))
        if sr != 16000:
            from torchaudio.transforms import Resample
            waveform = Resample(sr, 16000)(waveform)
            sr = 16000

        centroids, histories = {}, {}
        next_idx = 0

        def new_label():
            nonlocal next_idx
            label = f"spk_{next_idx}"
            next_idx += 1
            return label

        stitched = []
        for seg in sorted(raw, key=lambda x: x["start"]):
            start, end = seg["start"], seg["end"]
            start_sample, end_sample = int(start * sr), int(end * sr)
            if end_sample <= start_sample or start_sample >= waveform.size(1):
                stitched.append(seg); continue
            segment_wave = waveform[:, start_sample:end_sample]
            if segment_wave.numel() == 0:
                stitched.append(seg); continue
            if segment_wave.size(0) > 1:
                segment_wave = segment_wave.mean(dim=0, keepdim=True)
            with torch.no_grad():
                emb = model.encode_batch(segment_wave).squeeze()
            emb = F.normalize(emb.flatten(), p=2, dim=0)

            best_label, best_sim = None, -1.0
            for lbl, cent in centroids.items():
                sim = torch.dot(emb, cent).item()
                if sim > best_sim:
                    best_sim, best_label = sim, lbl

            if best_label and best_sim >= SPEAKER_STITCH_THRESHOLD:
                centroids[best_label] = F.normalize(
                    SPEAKER_STITCH_EMA_ALPHA * emb + (1 - SPEAKER_STITCH_EMA_ALPHA) * centroids[best_label],
                    p=2, dim=0
                )
                histories[best_label].append(emb)
                if len(histories[best_label]) > SPEAKER_STITCH_POOL_SIZE:
                    histories[best_label].pop(0)
                seg["speaker"] = best_label
            else:
                lbl = new_label()
                centroids[lbl] = emb
                histories[lbl] = [emb]
                seg["speaker"] = lbl
            stitched.append(seg)

        # optional merge centroids
        return stitched
    except Exception as e:
        logger.warning(f"[{upload_id}] stitching failed: {e}")
        return raw

@worker_process_init.connect
def preload_on_startup(**kwargs):
    if _HF_AVAILABLE:
        try:
            get_whisper_model()
        except Exception:
            pass
    if _PN_AVAILABLE:
        try:
            get_diarization_pipeline()
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5, device=0)
            logger.info("[INIT] Set CUDA memory fraction to 50%")
    except ImportError:
        pass

# --- Celery tasks ---

@app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] convert_to_wav_and_preview")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "processing_started"}))
    send_webhook_event("processing_started", upload_id, None)
    try:
        prepare_wav(upload_id)
    except Exception as e:
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        send_webhook_event("processing_failed", upload_id, None)
        return
    preview_transcribe.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] preview_transcribe")
    proc = prepare_preview_segment(upload_id)
    model = get_whisper_model()
    segments_gen, _ = model.transcribe(
        proc.stdout,
        word_timestamps=True,
        **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
    )
    proc.stdout.close(); proc.wait()
    segments = list(segments_gen)
    for s in segments:
        r.publish(
            f"progress:{upload_id}",
            json.dumps({"status": "preview_partial", "fragment": {"start": s.start, "end": s.end, "text": s.text}})
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

@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] transcribe_segments")
    wav, duration = prepare_wav(upload_id)
    if not _HF_AVAILABLE:
        send_webhook_event("processing_failed", upload_id, None)
        return

    model = get_whisper_model()
    raw_segs = []

    def _transcribe(source, offset=0.0):
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
        for s in segs:
            s.start += offset; s.end += offset
        return segs

    try:
        if duration <= settings.VAD_MAX_LENGTH_S:
            raw_segs = _transcribe(str(wav))
        else:
            chunks = math.ceil(duration / settings.CHUNK_LENGTH_S)
            offset = 0.0
            for idx in range(chunks):
                length = min(settings.CHUNK_LENGTH_S, duration - offset)
                p = subprocess.Popen(
                    ["ffmpeg","-y","-threads",str(settings.FFMPEG_THREADS),"-ss",str(offset),"-t",str(length),"-i",str(wav),"-f","wav","pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                try:
                    segs = _transcribe(p.stdout, offset)
                    raw_segs.extend(segs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        import torch; torch.cuda.empty_cache()
                        logger.error(f"OOM on chunk {idx+1}, retrying")
                        time.sleep(5)
                        segs = _transcribe(p.stdout, offset)
                        raw_segs.extend(segs)
                    else:
                        logger.error(f"Error chunk {idx+1}: {e}", exc_info=True)
                finally:
                    p.stdout.close(); p.wait()
                offset += length
    except Exception as e:
        import torch; torch.cuda.empty_cache()
        logger.error(f"transcribe_segments failed: {e}", exc_info=True)
        send_webhook_event("processing_failed", upload_id, None)
        return

    flat = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segs]
    flat.sort(key=lambda x: x["start"])
    sentences = group_into_sentences(flat)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript_original.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
    # Attempt polish but do not override sentences format
    try:
        raw_text = " ".join(s["text"] for s in sentences)
        _ = polish_with_gpt(raw_text)
    except Exception:
        pass
    # Always write list of sentences so front-end can parse
    (out / "transcript.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))
    send_webhook_event("transcription_completed", upload_id, {"transcript": sentences})

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    send_webhook_event("diarization_started", upload_id, None)

    wav, duration = prepare_wav(upload_id)
    try:
        chunk_limit = int(getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0))
    except Exception:
        chunk_limit = 0
    using_chunk = bool(chunk_limit and duration > chunk_limit)
    total = math.ceil(duration / chunk_limit) if using_chunk else 1

    if not _PN_AVAILABLE:
        send_webhook_event("processing_failed", upload_id, None)
        return

    pipeline = get_diarization_pipeline()
    raw = []

    try:
        if using_chunk:
            offset = 0.0
            for idx in range(total):
                length = min(chunk_limit, duration - offset)
                tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{idx}.wav"
                subprocess.run([
                    "ffmpeg","-y","-threads",str(max(1,settings.FFMPEG_THREADS//2)),
                    "-ss",str(offset),"-t",str(length),
                    "-i",str(wav),str(tmp)
                ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                try:
                    ann = pipeline(str(tmp))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        import torch; torch.cuda.empty_cache()
                        time.sleep(5)
                        ann = pipeline(str(tmp))
                    else:
                        raise
                for s, _, spk in ann.itertracks(yield_label=True):
                    raw.append({"start": float(s.start)+offset, "end": float(s.end)+offset, "speaker": spk})
                tmp.unlink(missing_ok=True)
                offset += length
        else:
            try:
                ann = pipeline(str(wav))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    import torch; torch.cuda.empty_cache()
                    time.sleep(5)
                    ann = pipeline(str(wav))
                else:
                    raise
            for s, _, spk in ann.itertracks(yield_label=True):
                raw.append({"start": float(s.start), "end": float(s.end), "speaker": spk})
    except Exception as e:
        import torch; torch.cuda.empty_cache()
        logger.error(f"diarize_full failed: {e}", exc_info=True)
        send_webhook_event("processing_failed", upload_id, None)
        return

    raw.sort(key=lambda x: x["start"])
    if SPEAKER_STITCH_ENABLED:
        raw = stitch_speakers(raw, wav, upload_id)

    # merge contiguous same-speaker segments
    diar = []
    buf = None
    for seg in raw:
        if buf and buf["speaker"] == seg["speaker"] and seg["start"] - buf["end"] < 0.1:
            buf["end"] = seg["end"]
        else:
            if buf:
                diar.append(buf)
            buf = dict(seg)
    if buf:
        diar.append(buf)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "diarization.json").write_text(json.dumps(diar, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarization_done", "segments": len(diar)}))
    send_webhook_event("diarization_completed", upload_id, {"diarization": diar})

@app.task(
    bind=True,
    name="deliver_webhook",
    queue="webhooks",
    max_retries=5,
    default_retry_delay=30,
)
def deliver_webhook(self, event_type: str, upload_id: str, data: Optional[Any]):
    send_webhook_event(event_type, upload_id, data)

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