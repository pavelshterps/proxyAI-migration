import os
import json
import logging
import time
import math
import subprocess
from pathlib import Path
from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from pydub import AudioSegment
from redis import Redis

from config.settings import settings
from config.celery import app
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        device  = settings.WHISPER_DEVICE.lower()
        compute = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(f"Compute '{compute}' unsupported on CPU; falling back to int8")
            compute = "int8"
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=device,
            compute_type=compute
        )
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _clustering_diarizer

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    device = settings.WHISPER_DEVICE.lower()
    try:
        if device == "cpu":
            opts = {}
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE
            get_whisper_model().transcribe(str(sample), **opts)
        else:
            get_vad().apply({"audio": str(sample)})
            get_clustering_diarizer().apply({"audio": str(sample)})
    except Exception:
        pass

@app.task(bind=True, name="tasks.preview_transcribe", queue="transcribe_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    r   = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    upl = Path(settings.UPLOAD_FOLDER)
    src = next(upl.glob(f"{upload_id}.*"), None)
    if not src:
        logger.error(f"[{correlation_id}] no source for {upload_id}")
        return

    # --- ensure WAV ---
    wav_path = str(src)
    if src.suffix.lower() != ".wav":
        wav_path = str(upl / f"{upload_id}.wav")
        convert_to_wav(src, wav_path)

    # --- extract preview chunk ---
    preview_wav = upl / f"{upload_id}_preview.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", "0",
        "-t", str(settings.PREVIEW_LENGTH_S),
        "-i", wav_path,
        str(preview_wav)
    ], check=True)

    # --- compute total chunks ---
    duration_str = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path
    ]).strip()
    full_len_s   = float(duration_str)
    total_chunks = max(1, math.ceil(full_len_s / settings.CHUNK_LENGTH_S))

    # --- transcribe preview ---
    model = get_whisper_model()
    opts  = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segs, _ = model.transcribe(str(preview_wav), word_timestamps=False, **opts)

    preview = {
        "text": "".join(s.text for s in segs),
        "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    }
    state = {
        "status": "preview_done",
        "preview": preview,
        "chunks_total": total_chunks,
        "chunks_done": 0,
        "diarize_requested": False
    }

    r.set(f"preview_result:{upload_id}", json.dumps(preview, ensure_ascii=False))
    r.set(f"progress:{upload_id}",    json.dumps(state,   ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state,   ensure_ascii=False))

    # --- split remainder ---
    split_audio.delay(upload_id, correlation_id)


# === далее идут split_audio, dispatch_transcription, transcribe_chunk,
#     collect_transcription, diarize_full, cleanup_old_uploads ===
#     (оставляем без изменений по логике, как в предыдущей версии)