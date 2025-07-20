import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
from celery.signals import worker_process_init
from redis import Redis

from config.settings import settings
from config.celery import celery_app

# Logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Webhook helper
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
    try:
        requests.post(url, json=payload, headers=headers, timeout=5)
    except Exception as e:
        logger.warning(f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} failed: {e}")

# Model flags
try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info("[INIT] faster-whisper available")
except ImportError as e:
    _HF_AVAILABLE = False
    logger.warning(f"[INIT] faster-whisper not available: {e}")

try:
    from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
    _PN_AVAILABLE = True
    logger.info("[INIT] pyannote.audio available")
except ImportError as e:
    _PN_AVAILABLE = False
    logger.warning(f"[INIT] pyannote.audio not available: {e}")

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model(model_override: str = None):
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(
        settings,
        "WHISPER_COMPUTE_TYPE",
        "float16" if device.startswith("cuda") else "int8"
    ).lower()

    if model_override:
        return WhisperModel(model_override, device=device, compute_type=compute)

    global _whisper_model
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        logger.info(f"[WHISPER] initializing model {model_id} on {device} ({compute})")
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
        logger.info(f"[WHISPER] model ready on {device} ({compute})")
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        logger.info("[VAD] loading VAD model")
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("[VAD] ready")
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        logger.info("[DIARIZER] loading diarizer pipeline")
        Path(settings.DIARIZER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("[DIARIZER] ready")
    return _clustering_diarizer

@worker_process_init.connect
def preload_on_startup(**kwargs):
    device = settings.WHISPER_DEVICE.lower()
    logger.info(f"[WARMUP] HF={_HF_AVAILABLE}, PN={_PN_AVAILABLE}, DEV={device}")
    if _HF_AVAILABLE:
        sample = Path(__file__).parent / "tests/fixtures/sample.wav"
        try:
            get_whisper_model().transcribe(
                str(sample),
                max_initial_timestamp=settings.PREVIEW_LENGTH_S
            )
            logger.info("[WARMUP] Whisper warmup ok")
        except Exception:
            logger.warning("[WARMUP] Whisper warmup failed")
    if _PN_AVAILABLE and device.startswith("cuda"):
        try:
            get_vad()
            get_clustering_diarizer()
            logger.info("[WARMUP] VAD & diarizer warmup ok")
        except Exception:
            logger.warning("[WARMUP] VAD/diarizer warmup failed")

# Audio utils
def probe_audio(src: Path) -> dict:
    res = subprocess.run(
        ["ffprobe","-v","error","-print_format","json","-show_format","-show_streams", str(src)],
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
    start = time.perf_counter()
    logger.info(f"[PREPARE] start for {upload_id}")
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    info = probe_audio(src)
    duration = info["duration"]
    if (src.suffix.lower() == ".wav"
        and info.get("codec_name") == "pcm_s16le"
        and info.get("sample_rate") == 16000
        and info.get("channels") == 1):
        if src != target:
            src.rename(target)
        logger.info(f"[PREPARE] WAV OK ({time.perf_counter()-start:.2f}s)")
        return target, duration

    subprocess.run(
        ["ffmpeg","-y","-threads",str(settings.FFMPEG_THREADS),
         "-i",str(src),"-acodec","pcm_s16le","-ac","1","-ar","16000",str(target)],
        check=True, stderr=subprocess.DEVNULL
    )
    logger.info(f"[PREPARE] converted ({time.perf_counter()-start:.2f}s)")
    return target, duration

# --- Celery tasks ---

@celery_app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    send_webhook_event("processing_started", upload_id, None)

    try:
        prepare_wav(upload_id)
    except Exception as e:
        logger.error(f"[{cid}] PREPARE ERROR: {e}", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        send_webhook_event("processing_failed", upload_id, None)
        return

    # fallback: если очередь preview_gpu длиннее threshold — на CPU, иначе на GPU-превью
    try:
        gpu_len = r.llen("preview_gpu") or 0
    except Exception:
        gpu_len = 0

    queue_name = (
        "transcribe_cpu"
        if gpu_len > settings.PREVIEW_GPU_QUEUE_THRESHOLD
        else "preview_gpu"
    )
    logger.info(f"[{cid}] enqueue PREVIEW -> {queue_name} (len={gpu_len})")
    from tasks import preview_transcribe
    preview_transcribe.apply_async((upload_id, correlation_id), queue=queue_name)

@celery_app.task(bind=True, queue="preview_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    try:
        logger.info(f"[{cid}] PREVIEW start for {upload_id}")
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        if not wav.exists():
            raise FileNotFoundError("WAV not found")

        proc = subprocess.Popen(
            ["ffmpeg","-y","-threads",str(settings.FFMPEG_THREADS//2 or 1),
             "-ss","0","-t",str(settings.PREVIEW_LENGTH_S),
             "-i",str(wav),"-f","wav","pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        segments_gen, _ = get_whisper_model().transcribe(
            proc.stdout, word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
        )
        proc.stdout.close(); proc.wait()
        segments = list(segments_gen)
        logger.info(f"[{cid}] got {len(segments)} segs")

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
            "timestamps": [{"start":s.start,"end":s.end,"text":s.text} for s in segments],
        }
        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preview_transcript.json").write_text(
            json.dumps(preview, ensure_ascii=False, indent=2)
        )
        r.publish(f"progress:{upload_id}", json.dumps({"status":"preview_done","preview":preview}))
        send_webhook_event("preview_completed", upload_id, {"preview": preview})

    except Exception as e:
        logger.error(f"[{cid}] PREVIEW ERROR: {e}", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        send_webhook_event("processing_failed", upload_id, None)
        return

    logger.info(f"[{cid}] enqueue TRANSCRIBE")
    from tasks import transcribe_segments
    transcribe_segments.delay(upload_id, correlation_id)

@celery_app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    try:
        logger.info(f"[{cid}] TRANSCRIBE start for {upload_id}")
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        if not wav.exists():
            raise FileNotFoundError("WAV not found")

        info = probe_audio(wav)
        duration = info["duration"]
        model = get_whisper_model()
        all_segs = []
        chunk_len = settings.CHUNK_LENGTH_S
        threads = settings.FFMPEG_THREADS

        if duration <= chunk_len:
            segs, _ = model.transcribe(
                str(wav), word_timestamps=True,
                **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
            )
            all_segs = list(segs)
        else:
            offset = 0.0
            while offset < duration:
                this_len = min(chunk_len, duration - offset)
                proc = subprocess.Popen(
                    ["ffmpeg","-y","-threads",str(threads),
                     "-ss",str(offset),"-t",str(this_len),
                     "-i",str(wav),"-f","wav","pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                seg_gen, _ = model.transcribe(
                    proc.stdout, word_timestamps=True,
                    **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
                )
                proc.stdout.close(); proc.wait()
                chunk_segs = list(seg_gen)
                for s in chunk_segs:
                    s.start += offset; s.end += offset
                all_segs.extend(chunk_segs)
                offset += this_len

        transcript_data = [{"start":s.start,"end":s.end,"text":s.text} for s in all_segs]
        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "transcript.json").write_text(
            json.dumps(transcript_data, ensure_ascii=False, indent=2)
        )
        r.publish(f"progress:{upload_id}", json.dumps({"status":"transcript_done"}))
        send_webhook_event("transcription_completed", upload_id, {"transcript": transcript_data})

    except Exception as e:
        logger.error(f"[{cid}] TRANSCRIBE ERROR: {e}", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        send_webhook_event("processing_failed", upload_id, None)

@celery_app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    send_webhook_event("diarization_started", upload_id, None)
    r.publish(f"progress:{upload_id}", json.dumps({"status":"diarize_started"}))

    if not _PN_AVAILABLE or not settings.WHISPER_DEVICE.lower().startswith("cuda"):
        err = "pyannote unavailable or not CUDA"
        logger.error(f"[{cid}] {err}")
        send_webhook_event("processing_failed", upload_id, None)
        return

    try:
        wav_path, _ = prepare_wav(upload_id)
        speech = get_vad().apply({"audio": str(wav_path)})
        ann = get_clustering_diarizer().apply({"audio": str(wav_path), "speech": speech})
        segs = [
            {"start": float(s.start), "end": float(s.end), "speaker": spk}
            for s, _, spk in ann.itertracks(yield_label=True)
        ]
        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diarization.json").write_text(
            json.dumps(segs, ensure_ascii=False, indent=2)
        )
        r.publish(f"progress:{upload_id}", json.dumps({"status":"diarization_done"}))
        send_webhook_event("diarization_completed", upload_id, {"diarization": segs})

    except Exception as e:
        logger.error(f"[{cid}] DIARIZE ERROR: {e}", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        send_webhook_event("processing_failed", upload_id, None)

@celery_app.task(bind=True, queue="transcribe_cpu")
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