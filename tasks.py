import json
import logging
import subprocess
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict

import requests
from redis import Redis
from celery.signals import worker_process_init
from celery import Task

from celery_app import app
from config.settings import settings

# --- Logger setup ---
logger = logging.getLogger(__name__)
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
    MAX_WORDS     = getattr(settings, "SENTENCE_MAX_WORDS", 50)

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
    return _speaker_embedding_model

def stitch_speakers(raw: List[Dict[str, Any]], wav: Path, upload_id: str) -> List[Dict[str, Any]]:
    if not SPEAKER_STITCH_ENABLED:
        return raw
    try:
        import torch
        import torchaudio
        model = get_speaker_embedding_model()
        # Load full wav once
        waveform, sr = torchaudio.load(str(wav))  # [channels, samples]
        if sr != 16000:
            from torchaudio.transforms import Resample
            resampler = Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # Normalize waveform to expected shape
        stitch_pool: Dict[str, List[Any]] = {}
        next_label_idx = 0

        def new_canonical_label():
            nonlocal next_label_idx
            label = f"spk_{next_label_idx}"
            next_label_idx += 1
            return label

        stitched: List[Dict[str, Any]] = []
        # Ensure deterministic order for incremental assignment
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
            # If multi-channel, average to mono
            if segment_waveform.size(0) > 1:
                segment_waveform = segment_waveform.mean(dim=0, keepdim=True)
            with torch.no_grad():
                emb = model.encode_batch(segment_waveform)  # [1, emb_dim]
            emb = emb.squeeze(0)  # [emb_dim]
            emb = emb / (emb.norm(p=2) + 1e-8)  # normalize

            assigned_label = None
            best_sim = -1.0
            candidate = None
            for canon_label, embeddings in stitch_pool.items():
                centroid = torch.stack(embeddings[-SPEAKER_STITCH_POOL_SIZE:]).mean(dim=0)
                centroid = centroid / (centroid.norm(p=2) + 1e-8)
                sim = torch.dot(emb, centroid).item()
                if sim > best_sim:
                    best_sim = sim
                    candidate = canon_label
            if candidate is not None and best_sim >= SPEAKER_STITCH_THRESHOLD:
                assigned_label = candidate
            else:
                assigned_label = new_canonical_label()
                stitch_pool[assigned_label] = []

            # Update pool
            stitch_pool[assigned_label].append(emb)
            if len(stitch_pool[assigned_label]) > SPEAKER_STITCH_POOL_SIZE:
                stitch_pool[assigned_label] = stitch_pool[assigned_label][-SPEAKER_STITCH_POOL_SIZE:]

            seg["speaker"] = assigned_label
            stitched.append(seg)
        return stitched
    except Exception as e:
        logger.warning(f"[{upload_id}] speaker stitching failed, falling back to original diarization labels: {e}")
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
    # если оба GPU заняты тяжёлыми тасками, переключаем на CPU
    try:
        inspector = app.control.inspect()
        active = inspector.active() or {}
        heavy = 0
        for node_tasks in active.values():
            for t in node_tasks:
                if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments"):
                    heavy += 1
        if heavy >= 2:
            logger.info(f"[{upload_id}] both GPUs busy (found {heavy} heavy tasks), falling back to CPU for transcription preview")
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
        logger.info(f"[{upload_id}] GPU memory reserved before transcription: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
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
            logger.info(f"[{upload_id}] GPUs appear busy ({heavy} heavy tasks), rescheduling transcription to CPU")
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
        logger.info(f"[{upload_id}] long audio ({duration:.1f}s) → chunking at {settings.CHUNK_LENGTH_S}s")
        offset = 0.0
        chunk_idx = 0
        while offset < duration:
            length = min(settings.CHUNK_LENGTH_S, duration - offset)
            logger.debug(f"[{upload_id}] chunk {chunk_idx} {offset:.1f}s→{offset+length:.1f}s")
            p = subprocess.Popen(
                [
                    "ffmpeg", "-y",
                    "-threads", str(settings.FFMPEG_THREADS),
                    "-ss", str(offset), "-t", str(length),
                    "-i", str(wav), "-f", "wav", "pipe:1"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            chunk_segs = _transcribe_with_vad(p.stdout, offset)
            p.stdout.close(); p.wait()

            raw_segs.extend(chunk_segs)
            offset += length
            chunk_idx += 1

    flat = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segs]
    flat.sort(key=lambda x: x["start"])
    sentences = group_into_sentences(flat)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
    logger.info(f"[{upload_id}] transcription completed ({len(sentences)} sentences)")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))
    deliver_webhook.delay("transcription_completed", upload_id, {"transcript": sentences})

    try:
        import torch
        logger.info(f"[{upload_id}] GPU memory reserved after transcription: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full started")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    deliver_webhook.delay("diarization_started", upload_id, None)

    try:
        import torch
        logger.info(f"[{upload_id}] GPU memory reserved before diarization: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

    wav, duration = prepare_wav(upload_id)

    raw_chunk_limit = getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0)
    try:
        chunk_limit = int(raw_chunk_limit)
    except Exception:
        logger.warning(f"[{upload_id}] invalid DIARIZATION_CHUNK_LENGTH_S={raw_chunk_limit!r}, falling back to 0")
        chunk_limit = 0

    logger.info(f"[{upload_id}] WAV prepared for diarization, duration={duration:.1f}s; chunk_limit={chunk_limit}")

    if not _PN_AVAILABLE:
        logger.error(f"[{upload_id}] pyannote.audio not available, aborting diarization")
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    pipeline = get_diarization_pipeline()
    raw: List[Dict[str, Any]] = []

    MAX_RETRIES_PER_CHUNK = 2

    if chunk_limit and duration > chunk_limit:
        offset = 0.0
        chunk_idx = 0
        while offset < duration:
            this_len = min(chunk_limit, duration - offset)
            logger.info(f"[{upload_id}] diarization chunk #{chunk_idx} start: {offset:.1f}s→{offset+this_len:.1f}s")
            tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{int(offset)}.wav"
            subprocess.run([
                "ffmpeg", "-y", "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
                "-ss", str(offset), "-t", str(this_len),
                "-i", str(wav), str(tmp)
            ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

            attempt = 0
            success = False
            while attempt < MAX_RETRIES_PER_CHUNK and not success:
                try:
                    ann = pipeline(str(tmp))
                    before = len(raw)
                    for s, _, spk in ann.itertracks(yield_label=True):
                        raw.append({
                            "start": float(s.start) + offset,
                            "end":   float(s.end)   + offset,
                            "speaker": spk
                        })
                    added = len(raw) - before
                    logger.info(f"[{upload_id}] diarization chunk #{chunk_idx} done: added {added} segments")
                    r.publish(f"progress:{upload_id}", json.dumps({
                        "status": "diarize_chunk_done",
                        "chunk_index": chunk_idx,
                        "offset": offset,
                        "length": this_len,
                        "added_segments": added,
                    }))
                    success = True
                except Exception as e:
                    attempt += 1
                    logger.warning(f"[{upload_id}] error in diarization chunk #{chunk_idx}, attempt {attempt}: {e}")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    time.sleep(5)

            if not success:
                logger.error(f"[{upload_id}] failed diarization chunk #{chunk_idx} after {MAX_RETRIES_PER_CHUNK} attempts")
                deliver_webhook.delay("processing_failed", upload_id, {"reason":"diarization_chunk_failure"})
                return

            tmp.unlink(missing_ok=True)
            offset += this_len
            chunk_idx += 1
    else:
        logger.info(f"[{upload_id}] Short audio or chunking disabled, single diarization pass")
        ann = pipeline(str(wav))
        for s, _, spk in ann.itertracks(yield_label=True):
            raw.append({
                "start": float(s.start),
                "end":   float(s.end),
                "speaker": spk
            })

    # сортируем
    raw.sort(key=lambda x: x["start"])

    # применяем stitching (в случае включённой фичи)
    if SPEAKER_STITCH_ENABLED:
        raw = stitch_speakers(raw, wav, upload_id)

    # агрегируем соседние сегменты одного спикера
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
    logger.info(f"[{upload_id}] diarization_done, total segments: {len(diar_sentences)}")
    try:
        import torch
        logger.info(f"[{upload_id}] GPU memory reserved after diarization: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
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
    cutoff = datetime.utcnow() - timedelta(days=age)
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