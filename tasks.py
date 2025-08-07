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

# --- полировка через GPT с безопасной обёрткой ---
def polish_with_gpt(text: str) -> str:
    try:
        resp = openai.chat.completions.create(
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
        return text  # fallback: исходный текст

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
        t0, t1 = max(0.0, t["start"] - pad), t["end"] + pad
        i = bisect_left(starts, t1)
        cands = [
            d for d in diar[max(0, i - 8): i + 8]
            if not (d["end"] <= t0 or d["start"] >= t1)
        ]
        if cands:
            best = max(cands, key=lambda d: max(0.0, min(d["end"], t1) - max(d["start"], t0)))
        else:
            best = nearest(i, t0, t1)
        out.append({**t, "speaker": best["speaker"]})
    return out

@worker_process_init.connect
def preload_on_startup(**kwargs):
    if _HF_AVAILABLE:
        # просто инициализируем whisper
        get_whisper_model()
    if _PN_AVAILABLE:
        get_diarization_pipeline()
    # ограничение фракции памяти CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5, device=0)
            logger.info("[INIT] CUDA memory fraction set to 50%")
    except ImportError:
        pass

# --- Celery tasks ---

@app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] convert_to_wav_and_preview received")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status": "processing_started"}))
    deliver_webhook.delay("processing_started", upload_id, None)
    try:
        prepare_wav(upload_id)
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
        heavy = sum(1 for tasks in active.values() for t in tasks if t["name"] in ("tasks.diarize_full","tasks.transcribe_segments"))
        if heavy >= 2:
            logger.info(f"[{upload_id}] GPUs busy → CPU preview")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] inspect failed → GPU preview")

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
        r.publish(f"progress:{upload_id}", json.dumps({
            "status":"preview_partial",
            "fragment":{"start":seg.start,"end":seg.end,"text":seg.text}
        }))
    preview = {
        "text":"".join(s.text for s in segments),
        "timestamps":[{"start":s.start,"end":s.end,"text":s.text} for s in segments],
    }
    out = Path(settings.RESULTS_FOLDER)/upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"preview_transcript.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status":"preview_done","preview":preview}))
    deliver_webhook.delay("preview_completed", upload_id, {"preview":preview})
    transcribe_segments.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] transcribe_segments received")
    try:
        import torch
        logger.info(f"[{upload_id}] pre-transcription GPU memory: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

    # CPU fallback inspection
    try:
        inspector = app.control.inspect()
        active = inspector.active() or {}
        heavy = sum(1 for tasks in active.values() for t in tasks if t["name"] in ("tasks.diarize_full","tasks.transcribe_segments") and t["name"]!="tasks.transcribe_segments")
        if heavy>=2 and self.request.delivery_info.get("routing_key")!="transcribe_cpu":
            logger.info(f"[{upload_id}] GPUs busy → CPU transcription")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] inspect failed for fallback")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav, duration = prepare_wav(upload_id)
    if not _HF_AVAILABLE:
        logger.error(f"[{upload_id}] whisper unavailable")
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    model = get_whisper_model()
    raw_segs: List[Any] = []

    def _transcribe_with_vad(source, offset=0.0):
        segs, _ = model.transcribe(
            source,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms":int(settings.SENTENCE_MAX_GAP_S*1000),
                "speech_pad_ms":200,
            },
            **({"language":settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
        )
        for s in segs:
            s.start += offset; s.end += offset
        return list(segs)

    if duration<=settings.VAD_MAX_LENGTH_S:
        raw_segs = _transcribe_with_vad(str(wav))
    else:
        total_chunks=math.ceil(duration/settings.CHUNK_LENGTH_S)
        processed_key=f"transcribe:processed_chunks:{upload_id}"
        processed={int(x) for x in r.smembers(processed_key)}
        offset=0.0; idx=0
        while offset<duration:
            if idx in processed:
                offset+=settings.CHUNK_LENGTH_S; idx+=1; continue
            length=min(settings.CHUNK_LENGTH_S, duration-offset)
            logger.info(f"[{upload_id}] chunk {idx+1}/{total_chunks}: {offset:.1f}→{offset+length:.1f}")
            try:
                p=subprocess.Popen(
                    ["ffmpeg","-y","-threads",str(settings.FFMPEG_THREADS),"-ss",str(offset),"-t",str(length),"-i",str(wav),"-f","wav","pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                chunk_segs=_transcribe_with_vad(p.stdout, offset)
                p.stdout.close(); p.wait()
                raw_segs.extend(chunk_segs)
                r.sadd(processed_key, idx)
            except Exception as e:
                logger.error(f"[{upload_id}] chunk {idx+1} error: {e}", exc_info=True)
                try: import torch; torch.cuda.empty_cache()
                except: pass
            finally:
                offset+=length; idx+=1
        r.delete(processed_key)

    flat=[{"start":s.start,"end":s.end,"text":s.text} for s in raw_segs]
    flat.sort(key=lambda x: x["start"])
    sentences=group_into_sentences(flat)

    out=Path(settings.RESULTS_FOLDER)/upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"transcript_original.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
    logger.info(f"[{upload_id}] saved original transcript ({len(sentences)} sentences)")

    try:
        raw_text=" ".join(s["text"] for s in sentences)
        polished=polish_with_gpt(raw_text)
        (out/"transcript.json").write_text(json.dumps({"text":polished}, ensure_ascii=False, indent=2))
        logger.info(f"[{upload_id}] saved polished transcript")
    except Exception as e:
        logger.warning(f"[{upload_id}] polishing failed: {e}")
        (out/"transcript.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))

    r.publish(f"progress:{upload_id}", json.dumps({"status":"transcript_done"}))
    deliver_webhook.delay("transcription_completed", upload_id, {"transcript": sentences})

    try:
        import torch
        logger.info(f"[{upload_id}] post-transcription GPU memory: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    r=Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full started")
    r.publish(f"progress:{upload_id}", json.dumps({"status":"diarize_started"}))
    deliver_webhook.delay("diarization_started", upload_id, None)

    try:
        import torch
        logger.info(f"[{upload_id}] pre-diarization GPU memory: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except: pass

    wav, duration=prepare_wav(upload_id)
    raw_chunk_limit=getattr(settings,"DIARIZATION_CHUNK_LENGTH_S",0)
    try:
        chunk_limit=int(raw_chunk_limit)
    except:
        logger.warning(f"[{upload_id}] invalid chunk limit {raw_chunk_limit}")
        chunk_limit=0

    using_chunking=bool(chunk_limit and duration>chunk_limit)
    logger.info(f"[{upload_id}] diarization setup: duration={duration:.1f}, chunk_limit={chunk_limit}, using_chunking={using_chunking}")

    if not _PN_AVAILABLE:
        logger.error(f"[{upload_id}] pyannote unavailable")
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    pipeline=get_diarization_pipeline()
    raw=[]

    if using_chunking:
        total_chunks=math.ceil(duration/chunk_limit)
        processed_key=f"diarize:processed_chunks:{upload_id}"
        processed={int(x) for x in r.smembers(processed_key)}
        offset=0.0; idx=0
        while offset<duration:
            if idx in processed:
                logger.info(f"[{upload_id}] skip diarize chunk {idx+1}/{total_chunks}")
                offset+=chunk_limit; idx+=1; continue
            length=min(chunk_limit, duration-offset)
            logger.info(f"[{upload_id}] diarize chunk {idx+1}/{total_chunks}: {offset:.1f}→{offset+length:.1f}")
            tmp=Path(settings.DIARIZER_CACHE_DIR)/f"{upload_id}_chunk_{idx}.wav"
            subprocess.run(
                ["ffmpeg","-y","-threads",str(max(1,settings.FFMPEG_THREADS//2)),"-ss",str(offset),"-t",str(length),"-i",str(wav),str(tmp)],
                check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )
            try:
                ann=pipeline(str(tmp))
                before=len(raw)
                for s,_,spk in ann.itertracks(yield_label=True):
                    raw.append({"start":float(s.start)+offset,"end":float(s.end)+offset,"speaker":spk})
                added=len(raw)-before
                logger.info(f"[{upload_id}] added {added} segments")
                r.sadd(processed_key, idx)
            except Exception as e:
                logger.error(f"[{upload_id}] diarize chunk {idx+1} error: {e}", exc_info=True)
                try: import torch; torch.cuda.empty_cache()
                except: pass
            finally:
                tmp.unlink(missing_ok=True)
                offset+=length; idx+=1
        r.delete(processed_key)
    else:
        logger.info(f"[{upload_id}] single-pass diarization")
        ann=pipeline(str(wav))
        for s,_,spk in ann.itertracks(yield_label=True):
            raw.append({"start":float(s.start),"end":float(s.end),"speaker":spk})

    raw.sort(key=lambda x: x["start"])
    if SPEAKER_STITCH_ENABLED and using_chunking:
        raw=stitch_speakers(raw, wav, upload_id)
    else:
        logger.debug(f"[{upload_id}] skipping stitching")

    diar_sentences=[]
    buf=None
    for seg in raw:
        if buf and buf["speaker"]==seg["speaker"] and seg["start"]-buf["end"]<0.1:
            buf["end"]=seg["end"]
        else:
            if buf: diar_sentences.append(buf)
            buf=dict(seg)
    if buf: diar_sentences.append(buf)

    out=Path(settings.RESULTS_FOLDER)/upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"diarization.json").write_text(json.dumps(diar_sentences, ensure_ascii=False, indent=2))
    logger.info(f"[{upload_id}] diarization_done: {len(diar_sentences)} segments")
    try:
        import torch
        logger.info(f"[{upload_id}] post-diarization GPU memory: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except: pass
    r.publish(f"progress:{upload_id}", json.dumps({"status":"diarization_done","segments":len(diar_sentences)}))
    deliver_webhook.delay("diarization_completed", upload_id, {"diarization":diar_sentences})

@app.task(
    bind=True,
    name="deliver_webhook",
    queue="webhooks",
    max_retries=5,
    default_retry_delay=30,
)
def deliver_webhook(self, event_type: str, upload_id: str, data: Optional[Any]):
    url=settings.WEBHOOK_URL; secret=settings.WEBHOOK_SECRET
    if not url or not secret: return
    payload={"event_type":event_type,"upload_id":upload_id,"timestamp":datetime.utcnow().replace(microsecond=0).isoformat()+"Z","data":data}
    headers={"Content-Type":"application/json","X-WebHook-Secret":secret}
    try:
        resp=requests.post(url,json=payload,headers=headers,timeout=(5,30)); code=resp.status_code
        if 200<=code<300 or code==405: logger.info(f"[WEBHOOK] {event_type} ok ({code})"); return
        if 400<=code<500: logger.error(f"[WEBHOOK] {event_type} returned {code}"); return
        raise Exception(f"Webhook {code}")
    except Exception as exc:
        logger.warning(f"[WEBHOOK] retrying: {exc}")
        raise self.retry(exc=exc)

@app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    age=settings.FILE_RETENTION_DAYS; deleted=0
    for base in (Path(settings.UPLOAD_FOLDER), Path(settings.RESULTS_FOLDER)):
        for p in base.glob("**/*"):
            try:
                if datetime.utcnow()-datetime.fromtimestamp(p.stat().st_mtime)>timedelta(days=age):
                    if p.is_dir(): p.rmdir()
                    else: p.unlink()
                    deleted+=1
            except: continue
    logger.info(f"[CLEANUP] deleted {deleted} files")