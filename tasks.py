# tasks.py

import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from celery.signals import worker_process_init
from redis import Redis

from config.settings import settings
from config.celery import celery_app

# --- Logger setup ---
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Check model availability ---
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


# --- Model loaders ---
def get_whisper_model(model_override: str = None):
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(
        settings, "WHISPER_COMPUTE_TYPE",
        "float16" if device.startswith("cuda") else "int8"
    ).lower()

    if model_override:
        return WhisperModel(model_override, device=device, compute_type=compute)

    global _whisper_model
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        try:
            path = download_model(model_id,
                                  cache_dir=settings.HUGGINGFACE_CACHE_DIR,
                                  local_files_only=(device == "cpu"))
        except Exception:
            path = model_id
        if device == "cpu" and compute in ("fp16", "float16"):
            compute = "int8"
        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
    return _whisper_model


# --- Audio prep & metadata ---
def probe_audio(src: Path) -> dict:
    res = subprocess.run([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(src)
    ], capture_output=True, text=True)
    data = {"duration": 0.0}
    try:
        j = json.loads(res.stdout)
        data["duration"] = float(j["format"].get("duration", 0.0))
        for s in j.get("streams", []):
            if s.get("codec_type") == "audio":
                data.update({
                    "codec_name": s.get("codec_name"),
                    "sample_rate": int(s.get("sample_rate", 0)),
                    "channels": int(s.get("channels", 0))
                })
                break
    except Exception:
        pass
    return data


def prepare_wav(upload_id: str) -> (Path, float):
    """
    Convert to 16k mono WAV only if needed.
    Returns (wav_path, duration).
    """
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = src.with_suffix(".wav")
    info = probe_audio(src)
    duration = info["duration"]

    if (src.suffix.lower() == ".wav" and
        info.get("codec_name") == "pcm_s16le" and
        info.get("sample_rate") == 16000 and
        info.get("channels") == 1):
        if src != target:
            src.rename(target)
        return target, duration

    threads = getattr(settings, "FFMPEG_THREADS", 2)
    subprocess.run([
        "ffmpeg", "-y",
        "-threads", str(threads),
        "-i", str(src),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        str(target)
    ], check=True, stderr=subprocess.DEVNULL)
    return target, duration


# --- Tasks ---
@celery_app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    """
    For short files (<= PREVIEW_LENGTH_S), do full transcription once and publish as preview;
    for longer, enqueue preview task.
    """
    cid = correlation_id or "?"
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    try:
        wav_path, duration = prepare_wav(upload_id)
    except Exception as e:
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        return

    # Short audio: single transcription used for both preview & full
    if duration <= settings.PREVIEW_LENGTH_S:
        model = get_whisper_model(settings.PREVIEW_WHISPER_MODEL) \
                if getattr(settings, "PREVIEW_WHISPER_MODEL", None) \
                else get_whisper_model()
        segments_gen, _ = model.transcribe(
            str(wav_path),
            max_initial_timestamp=settings.PREVIEW_LENGTH_S,
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        segments = list(segments_gen)
        preview = {
            "text": "".join(s.text for s in segments),
            "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        }
        r.publish(f"progress:{upload_id}", json.dumps({"status": "preview_done", "preview": preview}))
        from tasks import transcribe_segments
        transcribe_segments.delay(upload_id, cid)
        return

    # Longer audio: enqueue preview then full transcription
    from tasks import preview_transcribe
    preview_transcribe.delay(upload_id, cid)


@celery_app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    """
    Fast preview: first N seconds via ffmpeg → Whisper.
    """
    cid = correlation_id or "?"
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav_path = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav_path.exists():
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": "WAV not found"}))
        return

    # Single ffmpeg thread for preview
    cmd = [
        "ffmpeg", "-y", "-threads", "1",
        "-i", str(wav_path),
        "-ss", "0", "-t", str(settings.PREVIEW_LENGTH_S),
        "-f", "wav", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    model = get_whisper_model(settings.PREVIEW_WHISPER_MODEL) \
            if getattr(settings, "PREVIEW_WHISPER_MODEL", None) \
            else get_whisper_model()
    segments_gen, _ = model.transcribe(
        proc.stdout,
        word_timestamps=True,
        **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
    )
    proc.stdout.close()
    proc.wait()

    segments = list(segments_gen)
    for seg in segments:
        r.publish(f"progress:{upload_id}", json.dumps({
            "status": "preview_partial",
            "fragment": {"start": seg.start, "end": seg.end, "text": seg.text}
        }))

    preview = {
        "text": "".join(s.text for s in segments),
        "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    }
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "preview_transcript.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status": "preview_done", "preview": preview}))

    from tasks import transcribe_segments
    transcribe_segments.delay(upload_id, cid)


@celery_app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    """
    Full transcription: chunk if > CHUNK_LENGTH.
    """
    cid = correlation_id or "?"
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav_path = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav_path.exists():
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": "WAV not found"}))
        return

    info = probe_audio(wav_path)
    duration = info["duration"]
    chunk_len = getattr(settings, "CHUNK_LENGTH", 300)
    model = get_whisper_model()

    segments = []
    if duration <= chunk_len:
        segs, _ = model.transcribe(
            str(wav_path),
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        segments = list(segs)
    else:
        offset = 0.0
        while offset < duration:
            this_len = min(chunk_len, duration - offset)
            cmd = [
                "ffmpeg", "-y",
                "-threads", str(getattr(settings, "FFMPEG_THREADS", 2)),
                "-i", str(wav_path),
                "-ss", str(offset), "-t", str(this_len),
                "-f", "wav", "pipe:1"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            segs_gen, _ = model.transcribe(
                proc.stdout,
                word_timestamps=True,
                **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
            )
            proc.stdout.close()
            proc.wait()
            chunk_segs = list(segs_gen)
            for s in chunk_segs:
                s.start += offset
                s.end += offset
            segments.extend(chunk_segs)
            offset += this_len

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "transcript.json").write_text(json.dumps(
        [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        ensure_ascii=False, indent=2
    ))
    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))


@celery_app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    """
    Remove files older than FILE_RETENTION_DAYS.
    """
    age = settings.FILE_RETENTION_DAYS
    cutoff = datetime.utcnow() - timedelta(days=age)
    deleted = 0
    for base in (Path(settings.UPLOAD_FOLDER), Path(settings.RESULTS_FOLDER)):
        for p in base.rglob("*"):
            try:
                if datetime.utcnow() - datetime.fromtimestamp(p.stat().st_mtime) > timedelta(days=age):
                    if p.is_dir(): p.rmdir()
                    else: p.unlink()
                    deleted += 1
            except Exception:
                continue
    logger.info(f"[{datetime.utcnow().isoformat()}] [CLEANUP] deleted {deleted} old files")