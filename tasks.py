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
    '%(asctime)s %(levelname)s [%(name)s] %(messagefmt)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --- Check availability of fast models ---
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
    """
    Return WhisperModel, optionally overridden for preview.
    """
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(
        settings, "WHISPER_COMPUTE_TYPE",
        "float16" if device.startswith("cuda") else "int8"
    ).lower()

    if model_override:
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] override model "
                    f"{model_override} on {device} ({compute})")
        return WhisperModel(model_override, device=device, compute_type=compute)

    global _whisper_model
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] initializing model "
                    f"{model_id} on {device} ({compute})")
        try:
            path = download_model(
                model_id,
                cache_dir=settings.HUGGINGFACE_CACHE_DIR,
                local_files_only=(device == "cpu")
            )
        except Exception:
            path = model_id
        if device == "cpu" and compute in ("fp16", "float16"):
            compute = "int8"
        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] model ready "
                    f"on {device} ({compute})")
    return _whisper_model


def get_vad():
    """
    Load VAD model on first call.
    """
    global _vad
    if _vad is None:
        logger.info(f"[{datetime.utcnow().isoformat()}] [VAD] loading VAD model")
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"[{datetime.utcnow().isoformat()}] [VAD] ready")
    return _vad


def get_clustering_diarizer():
    """
    Load speaker diarization pipeline on first call.
    """
    global _clustering_diarizer
    if _clustering_diarizer is None:
        logger.info(f"[{datetime.utcnow().isoformat()}] [DIARIZER] loading diarizer pipeline")
        Path(settings.DIARIZER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info(f"[{datetime.utcnow().isoformat()}] [DIARIZER] ready")
    return _clustering_diarizer


@worker_process_init.connect
def preload_on_startup(**kwargs):
    """
    Warm up models on worker start.
    """
    device = settings.WHISPER_DEVICE.lower()
    logger.info(f"[{datetime.utcnow().isoformat()}] [WARMUP] starting "
                f"(HF={_HF_AVAILABLE}, PN={_PN_AVAILABLE}, DEV={device})")
    if _HF_AVAILABLE:
        sample = Path(__file__).parent / "tests/fixtures/sample.wav"
        try:
            get_whisper_model().transcribe(
                str(sample),
                max_initial_timestamp=settings.PREVIEW_LENGTH_S
            )
            logger.info(f"[{datetime.utcnow().isoformat()}] [WARMUP] Whisper warmup ok")
        except Exception:
            logger.warning(f"[{datetime.utcnow().isoformat()}] [WARMUP] Whisper warmup failed")
    if _PN_AVAILABLE and device.startswith("cuda"):
        try:
            get_vad()
            get_clustering_diarizer()
            logger.info(f"[{datetime.utcnow().isoformat()}] [WARMUP] VAD & diarizer warmup ok")
        except Exception:
            logger.warning(f"[{datetime.utcnow().isoformat()}] [WARMUP] VAD/diarizer warmup failed")


# --- Audio preparation & metadata ---
def probe_audio(src: Path) -> dict:
    """
    Single ffprobe call: returns duration and first audio stream params.
    """
    res = subprocess.run([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(src)
    ], capture_output=True, text=True)
    info = {"duration": 0.0}
    try:
        j = json.loads(res.stdout)
        info["duration"] = float(j["format"].get("duration", 0.0))
        for s in j.get("streams", []):
            if s.get("codec_type") == "audio":
                info.update({
                    "codec_name": s.get("codec_name"),
                    "sample_rate": int(s.get("sample_rate", 0)),
                    "channels": int(s.get("channels", 0))
                })
                break
    except Exception:
        pass
    return info


def prepare_wav(upload_id: str) -> (Path, float):
    """
    Convert to mono WAV 16k if needed. Returns (wav_path, duration_s).
    """
    start = time.perf_counter()
    logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] start for {upload_id}")

    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    info = probe_audio(src)
    duration = info["duration"]

    # already correct WAV?
    if (src.suffix.lower() == ".wav"
            and info.get("codec_name") == "pcm_s16le"
            and info.get("sample_rate") == 16000
            and info.get("channels") == 1):
        if src != target:
            src.rename(target)
        logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] WAV OK, renamed "
                    f"({time.perf_counter() - start:.2f}s)")
        return target, duration

    # convert
    threads = getattr(settings, "FFMPEG_THREADS", 4)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-threads", str(threads),
            "-i", str(src),
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            str(target)
        ], check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        logger.error(f"[{datetime.utcnow().isoformat()}] [PREPARE] ffmpeg failed", exc_info=True)
        raise
    logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] converted "
                f"({time.perf_counter() - start:.2f}s)")
    return target, duration


# --- Tasks ---
@celery_app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    """
    Prepare WAV, then choose: full (<=PREVIEW_LENGTH_S) or preview+full.
    """
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] CONVERT start for {upload_id}")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    try:
        wav_path, duration = prepare_wav(upload_id)
    except Exception as e:
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        return

    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] duration={duration:.1f}s")

    if duration <= settings.PREVIEW_LENGTH_S:
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] small audio, direct full transcription")
        from tasks import transcribe_segments
        transcribe_segments.delay(upload_id, correlation_id)
    else:
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] enqueue preview")
        from tasks import preview_transcribe
        preview_transcribe.delay(upload_id, correlation_id)

    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] CONVERT done")


@celery_app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    """
    Fast preview via WhisperModel(max_initial_timestamp), then enqueue full.
    """
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] PREVIEW start for {upload_id}")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav.exists():
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":"WAV not found"}))
        return

    model = get_whisper_model(settings.PREVIEW_WHISPER_MODEL) \
            if getattr(settings, "PREVIEW_WHISPER_MODEL", None) else get_whisper_model()

    start = time.perf_counter()
    try:
        segments_gen, _ = model.transcribe(
            str(wav),
            max_initial_timestamp=settings.PREVIEW_LENGTH_S,
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] PREVIEW error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        return

    segments = list(segments_gen)
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] got {len(segments)} preview "
                f"segments ({time.perf_counter()-start:.2f}s)")

    for seg in segments:
        r.publish(f"progress:{upload_id}", json.dumps({
            "status":"preview_partial",
            "fragment":{"start":seg.start,"end":seg.end,"text":seg.text}
        }))

    preview = {
        "text": "".join(s.text for s in segments),
        "timestamps":[{"start":s.start,"end":s.end,"text":s.text} for s in segments]
    }
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"preview_transcript.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status":"preview_done","preview":preview}))
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] PREVIEW done")

    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] enqueue full transcription")
    from tasks import transcribe_segments
    transcribe_segments.delay(upload_id, correlation_id)


@celery_app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    """
    Full transcription of WAV, chunked if long.
    """
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] TRANSCRIBE start for {upload_id}")
    if not _HF_AVAILABLE:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] Whisper unavailable, skipping")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav.exists():
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":"WAV not found"}))
        return

    info = probe_audio(wav)
    duration = info["duration"]
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] duration={duration:.1f}s")

    model = get_whisper_model()
    all_segs = []
    start = time.perf_counter()
    chunk_len = getattr(settings, "CHUNK_LENGTH", 300)
    threads = getattr(settings, "FFMPEG_THREADS", 4)

    if duration <= chunk_len:
        segs, _ = model.transcribe(
            str(wav),
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        all_segs = list(segs)
    else:
        offset = 0.0
        while offset < duration:
            this_len = min(chunk_len, duration - offset)
            logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] chunk offset={offset}s len={this_len}s")
            proc = subprocess.Popen([
                "ffmpeg", "-y", "-threads", str(threads),
                "-i", str(wav),
                "-ss", str(offset), "-t", str(this_len),
                "-f", "wav", "pipe:1"
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            seg_gen, _ = model.transcribe(
                proc.stdout,
                word_timestamps=True,
                **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
            )
            proc.stdout.close()
            proc.wait()
            chunk_segs = list(seg_gen)
            for s in chunk_segs:
                s.start += offset
                s.end += offset
            all_segs.extend(chunk_segs)
            offset += this_len

    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] got {len(all_segs)} segments "
                f"({time.perf_counter()-start:.2f}s)")
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"transcript.json").write_text(json.dumps(
        [{"start":s.start,"end":s.end,"text":s.text} for s in all_segs],
        ensure_ascii=False, indent=2
    ))
    r.publish(f"progress:{upload_id}", json.dumps({"status":"transcript_done"}))
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] TRANSCRIBE done")


@celery_app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    """
    Speaker diarization on GPU.
    """
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] DIARIZE start for {upload_id}")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status":"diarize_started"}))

    if not _PN_AVAILABLE or not settings.WHISPER_DEVICE.lower().startswith("cuda"):
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] pyannote unavailable or not CUDA, skipping")
        return

    try:
        wav_path, _ = prepare_wav(upload_id)
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] VAD apply")
        speech = get_vad().apply({"audio": str(wav_path)})
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] SpeakerDiarization apply")
        ann = get_clustering_diarizer().apply({"audio": str(wav_path), "speech": speech})
        segs = [
            {"start": float(s.start), "end": float(s.end), "speaker": spk}
            for s, _, spk in ann.itertracks(yield_label=True)
        ]
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] got {len(segs)} diarization segments")
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] DIARIZE error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        return

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out/"diarization.json").write_text(json.dumps(segs, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status":"diarization_done"}))
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] DIARIZE done")


@celery_app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    """
    Remove files older than retention period.
    """
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
    logger.info(f"[{datetime.utcnow().isoformat()}] [CLEANUP] deleted {deleted} old files")