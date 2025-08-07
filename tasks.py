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
    except Exception as e:
        logger.warning(f"[polish_with_gpt] OpenAI error: {e}")
        # fallback: вернуть оригинал
        return text

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

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=(5, 30))
        except requests.RequestException as e:
            logger.warning(f"[WEBHOOK] network error on {attempt}/{max_attempts} for {upload_id}: {e}")
        else:
            code = resp.status_code
            if 200 <= code < 300 or code == 405:
                logger.info(f"[WEBHOOK] {event_type} succeeded for {upload_id} ({code})")
                return
            if 400 <= code < 500:
                logger.error(f"[WEBHOOK] {event_type} returned {code} for {upload_id}, aborting")
                return
            logger.warning(f"[WEBHOOK] {event_type} returned {code}, retrying ({attempt}/{max_attempts})")
        time.sleep(30)
    logger.error(f"[WEBHOOK] {event_type} failed after {max_attempts} attempts for {upload_id}")

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
        best = (
            max(cands, key=lambda d: max(0.0, min(d["end"], t1) - max(d["start"], t0)))
            if cands else nearest(i, t0, t1)
        )
        out.append({**t, "speaker": best["speaker"]})
    return out

def get_whisper_model(model_override: str = None):
    global _whisper_model
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(settings, "WHISPER_COMPUTE_TYPE",
                      "float16" if device.startswith("cuda") else "int8").lower()
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
        logger.info("[STITCH] loaded speaker embedding model from speechbrain/spkrec-ecapa-voxceleb")
    return _speaker_embedding_model

def stitch_speakers(raw: List[Dict[str, Any]], wav: Path, upload_id: str) -> List[Dict[str, Any]]:
    if not SPEAKER_STITCH_ENABLED:
        return raw

    unique_orig = set(seg.get("speaker") for seg in raw)
    if len(unique_orig) <= 1:
        logger.debug(f"[{upload_id}] only one original speaker, skipping stitching")
        return raw

    try:
        import torch
        import torchaudio
        import torch.nn.functional as F

        model = get_speaker_embedding_model()

        waveform, sr = torchaudio.load(str(wav))
        if sr != 16000:
            from torchaudio.transforms import Resample
            waveform = Resample(sr, 16000)(waveform)
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
        for seg in sorted(raw, key=lambda x: x["start"]):
            start, end = seg["start"], seg["end"]
            if end <= start:
                stitched.append(seg)
                continue
            s_i, e_i = int(start * sr), int(end * sr)
            if e_i <= s_i or s_i >= waveform.size(1):
                stitched.append(seg)
                continue
            chunk = waveform[:, s_i:e_i]
            if chunk.numel() == 0:
                stitched.append(seg); continue
            if chunk.size(0) > 1:
                chunk = chunk.mean(dim=0, keepdim=True)
            with torch.no_grad():
                emb = model.encode_batch(chunk).squeeze()
            emb = F.normalize(emb.flatten() if emb.ndim>1 else emb, p=2, dim=0)

            assigned, best_sim = None, -1.0
            for label, centroid in stitch_centroids.items():
                sim = torch.dot(emb, centroid).item()
                if sim > best_sim:
                    best_sim, assigned = sim, label

            if assigned and best_sim >= SPEAKER_STITCH_THRESHOLD:
                stitch_centroids[assigned] = F.normalize(
                    SPEAKER_STITCH_EMA_ALPHA*emb + (1-SPEAKER_STITCH_EMA_ALPHA)*stitch_centroids[assigned],
                    p=2, dim=0
                )
                hist = stitch_histories[assigned]
                hist.append(emb); hist[:] = hist[-SPEAKER_STITCH_POOL_SIZE:]
                logger.debug(f"[{upload_id}] reused speaker {assigned} (sim={best_sim:.3f})")
            else:
                assigned = new_canonical_label()
                stitch_centroids[assigned] = emb
                stitch_histories[assigned] = [emb]
                logger.debug(f"[{upload_id}] created speaker {assigned}")

            stitched.append({**seg, "speaker": assigned})

        # merge close labels
        centroids = {l: torch.stack(h).mean(dim=0).normalize(p=2,dim=0) for l,h in stitch_histories.items()}
        adj = {l:set() for l in centroids}
        labels = list(centroids)
        for i in range(len(labels)):
            for j in range(i+1,len(labels)):
                if torch.dot(centroids[labels[i]],centroids[labels[j]]).item() >= SPEAKER_STITCH_MERGE_THRESHOLD:
                    adj[labels[i]].add(labels[j]); adj[labels[j]].add(labels[i])
        visited, merge_map = set(), {}
        for l in adj:
            if l in visited: continue
            comp = []
            stack=[l]
            while stack:
                x=stack.pop()
                if x in visited: continue
                visited.add(x); comp.append(x)
                stack+=list(adj[x]-visited)
            if len(comp)>1:
                rep=sorted(comp)[0]
                for x in comp: merge_map[x]=rep
        if merge_map:
            for seg in stitched:
                if seg["speaker"] in merge_map:
                    seg["speaker"]=merge_map[seg["speaker"]]

        return stitched

    except Exception as e:
        logger.warning(f"[{upload_id}] speaker stitching failed: {e}")
        return raw

@worker_process_init.connect
def preload_on_startup(**kwargs):
    if _HF_AVAILABLE:
        get_whisper_model()
    if _PN_AVAILABLE:
        get_diarization_pipeline()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5, device=0)
            logger.info("[INIT] Set per-process CUDA memory to 50%")
    except ImportError:
        pass

# --- Celery tasks ---

@app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] convert_to_wav_and_preview received")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status":"processing_started"}))
    deliver_webhook.delay("processing_started", upload_id, None)

    try:
        prepare_wav(upload_id)
    except Exception as e:
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    preview_transcribe.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] preview_transcribe received")
    try:
        active = app.control.inspect().active() or {}
        heavy = sum(1 for tasks in active.values() for t in tasks if t["name"] in ("tasks.diarize_full","tasks.transcribe_segments"))
        if heavy>=2:
            logger.info(f"[{upload_id}] GPUs busy ({heavy}), falling back CPU preview")
            transcribe_segments.apply_async((upload_id,correlation_id),queue="transcribe_cpu")
            return
    except Exception:
        logger.warning("inspect failed, GPU preview")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    proc = prepare_preview_segment(upload_id)
    model = get_whisper_model()
    try:
        segments_gen,_ = model.transcribe(proc.stdout, word_timestamps=True,
            **({"language":settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}))
    finally:
        proc.stdout.close(); proc.wait()
    segments=list(segments_gen)
    for s in segments:
        r.publish(f"progress:{upload_id}", json.dumps({"status":"preview_partial","fragment":{"start":s.start,"end":s.end,"text":s.text}}))
    preview={"text":"".join(s.text for s in segments),"timestamps":[{"start":s.start,"end":s.end,"text":s.text} for s in segments]}
    out=Path(settings.RESULTS_FOLDER)/upload_id; out.mkdir(parents=True,exist_ok=True)
    (out/"preview_transcript.json").write_text(json.dumps(preview,ensure_ascii=False,indent=2))
    r.publish(f"progress:{upload_id}",json.dumps({"status":"preview_done","preview":preview}))
    deliver_webhook.delay("preview_completed",upload_id,{"preview":preview})
    transcribe_segments.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] transcribe_segments received")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav, duration = prepare_wav(upload_id)
    if not _HF_AVAILABLE:
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    model = get_whisper_model()
    raw_segs=[]

    def _transcribe_with_vad(src, offset=0.0):
        try:
            segs,_=model.transcribe(src, word_timestamps=True, vad_filter=True,
                vad_parameters={"min_silence_duration_ms":int(settings.SENTENCE_MAX_GAP_S*1000),"speech_pad_ms":200},
                **({"language":settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}))
            for s in segs:
                s.start+=offset; s.end+=offset
            return segs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch; torch.cuda.empty_cache()
                raise
            else:
                raise

    try:
        if duration<=settings.VAD_MAX_LENGTH_S:
            raw_segs=_transcribe_with_vad(str(wav))
        else:
            chunks=math.ceil(duration/settings.CHUNK_LENGTH_S)
            offset=0; idx=0
            while offset<duration:
                length=min(settings.CHUNK_LENGTH_S,duration-offset)
                p=subprocess.Popen(["ffmpeg","-y","-threads",str(settings.FFMPEG_THREADS),
                                     "-ss",str(offset),"-t",str(length),"-i",str(wav),
                                     "-f","wav","pipe:1"],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
                try:
                    segs=_transcribe_with_vad(p.stdout,offset)
                    raw_segs.extend(segs)
                except RuntimeError:
                    logger.error(f"[{upload_id}] CUDA OOM in chunk {idx+1}, retry CPU")
                    transcribe_segments.apply_async((upload_id,correlation_id),queue="transcribe_cpu")
                    return
                finally:
                    p.stdout.close(); p.wait()
                offset+=length; idx+=1
    except Exception as e:
        logger.error(f"[{upload_id}] transcription error: {e}",exc_info=True)
        deliver_webhook.delay("processing_failed",upload_id,None)
        return

    flat=[{"start":s.start,"end":s.end,"text":s.text} for s in raw_segs]
    flat.sort(key=lambda x:x["start"])
    sentences=group_into_sentences(flat)

    out=Path(settings.RESULTS_FOLDER)/upload_id; out.mkdir(parents=True,exist_ok=True)
    (out/"transcript_original.json").write_text(json.dumps(sentences,ensure_ascii=False,indent=2))

    # полировка
    polished=polish_with_gpt(" ".join(s["text"] for s in sentences))
    (out/"transcript.json").write_text(json.dumps({"text":polished},ensure_ascii=False,indent=2))

    r.publish(f"progress:{upload_id}",json.dumps({"status":"transcript_done"}))
    deliver_webhook.delay("transcription_completed",upload_id,{"transcript":sentences})

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] diarize_full started")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}",json.dumps({"status":"diarize_started"}))
    deliver_webhook.delay("diarization_started",upload_id,None)

    wav,duration=prepare_wav(upload_id)
    try:
        chunk_limit=int(getattr(settings,"DIARIZATION_CHUNK_LENGTH_S",0))
    except ValueError:
        chunk_limit=0
    using_chunk=bool(chunk_limit and duration>chunk_limit)
    total_chunks=math.ceil(duration/chunk_limit) if using_chunk else 1

    if not _PN_AVAILABLE:
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    pipeline=get_diarization_pipeline()
    raw=[]

    if using_chunk:
        offset=0; idx=0
        while offset<duration:
            this_len=min(chunk_limit,duration-offset)
            tmp=Path(settings.DIARIZER_CACHE_DIR)/f"{upload_id}_chunk_{idx}.wav"
            subprocess.run(["ffmpeg","-y","-threads",str(max(1,settings.FFMPEG_THREADS//2)),
                            "-ss",str(offset),"-t",str(this_len),"-i",str(wav),str(tmp)],
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                ann=None
                try:
                    ann=pipeline(str(tmp))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        import torch; torch.cuda.empty_cache()
                        time.sleep(5)
                        ann=pipeline(str(tmp))
                    else:
                        raise
                for s,_,spk in ann.itertracks(yield_label=True):
                    raw.append({"start":float(s.start)+offset,"end":float(s.end)+offset,"speaker":spk})
            except Exception as e:
                logger.error(f"[{upload_id}] diarize chunk error: {e}",exc_info=True)
                import torch; torch.cuda.empty_cache()
            finally:
                tmp.unlink(missing_ok=True)
                offset+=this_len; idx+=1
    else:
        ann=None
        try:
            ann=pipeline(str(wav))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch; torch.cuda.empty_cache()
                time.sleep(5)
                ann=pipeline(str(wav))
            else:
                raise
        for s,_,spk in ann.itertracks(yield_label=True):
            raw.append({"start":float(s.start),"end":float(s.end),"speaker":spk})

    raw.sort(key=lambda x:x["start"])
    if SPEAKER_STITCH_ENABLED and using_chunk:
        raw=stitch_speakers(raw, wav, upload_id)

    diar_sentences=[]
    buf=None
    for seg in raw:
        if buf and buf["speaker"]==seg["speaker"] and seg["start"]-buf["end"]<0.1:
            buf["end"]=seg["end"]
        else:
            if buf: diar_sentences.append(buf)
            buf=dict(seg)
    if buf: diar_sentences.append(buf)

    out=Path(settings.RESULTS_FOLDER)/upload_id; out.mkdir(parents=True,exist_ok=True)
    (out/"diarization.json").write_text(json.dumps(diar_sentences,ensure_ascii=False,indent=2))
    r.publish(f"progress:{upload_id}",json.dumps({"status":"diarization_done","segments":len(diar_sentences)}))
    deliver_webhook.delay("diarization_completed",upload_id,{"diarization":diar_sentences})

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
    payload={"event_type":event_type,"upload_id":upload_id,
             "timestamp":datetime.utcnow().replace(microsecond=0).isoformat()+"Z","data":data}
    headers={"Content-Type":"application/json","X-WebHook-Secret":secret}
    try:
        resp=requests.post(url,json=payload,headers=headers,timeout=(5,30))
        code=resp.status_code
        if 200<=code<300 or code==405:
            return
        if 400<=code<500:
            return
        raise Exception(f"Webhook returned {code}")
    except Exception as exc:
        raise self.retry(exc=exc)

@app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    age=settings.FILE_RETENTION_DAYS; deleted=0
    for base in (Path(settings.UPLOAD_FOLDER),Path(settings.RESULTS_FOLDER)):
        for p in base.glob("**/*"):
            try:
                if datetime.utcnow()-datetime.fromtimestamp(p.stat().st_mtime)>timedelta(days=age):
                    if p.is_dir(): p.rmdir()
                    else: p.unlink()
                    deleted+=1
            except Exception:
                continue
    logger.info(f"[CLEANUP] deleted {deleted} old files")