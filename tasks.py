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
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(
        settings, "WHISPER_COMPUTE_TYPE",
        "float16" if device.startswith("cuda") else "int8"
    ).lower()

    if model_override:
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] override model {model_override} on {device} ({compute})")
        return WhisperModel(model_override, device=device, compute_type=compute)

    global _whisper_model
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] initializing model {model_id} on {device} ({compute})")
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
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] model ready on {device} ({compute})")
    return _whisper_model

def get_vad():
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
    device = settings.WHISPER_DEVICE.lower()
    logger.info(f"[{datetime.utcnow().isoformat()}] [WARMUP] starting (HF={_HF_AVAILABLE}, PN={_PN_AVAILABLE}, DEV={device})")
    if _HF_AVAILABLE:
        sample = Path(__file__).parent / "tests/fixtures/sample.wav"
        try:
            get_whisper_model().transcribe(
                str(sample), max_initial_timestamp=settings.PREVIEW_LENGTH_S
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

# --- Audio prep & metadata ---
def probe_audio(src: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(src)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
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
    start = time.perf_counter()
    logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] start for {upload_id}")
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    info = probe_audio(src)
    duration = info["duration"]
    if src.suffix.lower() == ".wav" and \
       info.get("codec_name") == "pcm_s16le" and \
       info.get("sample_rate") == 16000 and \
       info.get("channels") == 1:
        if src != target:
            src.rename(target)
        elapsed = time.perf_counter() - start
        logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] WAV OK, renamed ({elapsed:.2f}s)")
        return target, duration
    threads = getattr(settings, "FFMPEG_THREADS", 2)
    subprocess.run([
        "ffmpeg", "-y",
        "-threads", str(threads),
        "-i", str(src),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        str(target)
    ], check=True, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] converted ({elapsed:.2f}s)")
    return target, duration

# --- Tasks ---
@celery_app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
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
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] short audio, enqueue full transcription")
        from tasks import transcribe_segments
        transcribe_segments.delay(upload_id, correlation_id)
    else:
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] enqueue PREVIEW")
        from tasks import preview_transcribe
        preview_transcribe.delay(upload_id, correlation_id)
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] CONVERT done")