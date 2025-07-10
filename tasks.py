# tasks.py
import os
import json
import logging
import time
from pathlib import Path
from celery import Celery, shared_task
from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from utils.audio import convert_to_wav
from pydub import AudioSegment
from redis import Redis

from config.settings import settings

app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# обе задачи на GPU-очередь
app.conf.task_default_queue = "preprocess_gpu"
app.conf.task_routes = {
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
    "tasks.diarize_full":        {"queue": "preprocess_gpu"},
}

logger = logging.getLogger(__name__)
_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model(device: str):
    global _whisper_model
    if _whisper_model is None or _whisper_model.device != device:
        model_path = settings.WHISPER_MODEL_PATH
        compute = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(f"Compute '{compute}' not on CPU, fallback to int8"); compute = "int8"
        logger.info(f"Loading WhisperModel (path={model_path}, device={device}, compute={compute})")
        _whisper_model = WhisperModel(model_path, device=device, compute_type=compute)
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_vad():
    global _vad
    if not _vad:
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH, use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("VAD pipeline loaded")
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if not _clustering_diarizer:
        cache = settings.DIARIZER_CACHE_DIR; os.makedirs(cache, exist_ok=True)
        logger.info(f"Loading clustering diarizer into '{cache}'")
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE, cache_dir=cache, use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("Clustering-based diarizer loaded")
    return _clustering_diarizer

@worker_process_init.connect
def preload(**kwargs):
    sample = Path(__file__).parent / "tests/fixtures/sample.wav"
    device = settings.WHISPER_DEVICE.lower()
    if device == "cuda":
        try:
            get_vad().apply({"audio": str(sample)})
            get_clustering_diarizer().apply({"audio": str(sample)})
            logger.info("✅ Warm-up VAD+diarizer (GPU)")
        except Exception as e:
            logger.warning(f"Warm-up GPU pipelines failed: {e}")
    else:
        try:
            get_whisper_model(device).transcribe(str(sample))
            logger.info("✅ Warm-up Whisper (CPU)")
        except Exception as e:
            logger.warning(f"Warm-up Whisper (CPU) failed: {e}")

@shared_task(bind=True, name="tasks.transcribe_segments")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    try:
        # найти исходник
        folder = Path(settings.UPLOAD_FOLDER)
        files = list(folder.glob(f"{upload_id}.*"))
        if not files:
            raise FileNotFoundError(f"No source for {upload_id}")
        upload_path = files[0]
        ext = upload_path.suffix
        wav_path = folder / f"{upload_id}.wav"

        # конвертация
        if ext.lower() != ".wav":
            convert_to_wav(upload_path, wav_path, sample_rate=16000, channels=1)
            src = wav_path
        else:
            src = upload_path

        dst = Path(settings.RESULTS_FOLDER) / upload_id
        dst.mkdir(exist_ok=True, parents=True)
        adapter.info(f"Starting transcription for '{src}'")

        whisper = get_whisper_model(settings.WHISPER_DEVICE.lower())
        segments, _ = whisper.transcribe(str(src),
                                         beam_size=settings.WHISPER_BATCH_SIZE,
                                         word_timestamps=True,
                                         language=settings.WHISPER_LANGUAGE or None)

        transcript = [
            {"segment": i, "start": seg.start, "end": seg.end, "text": seg.text}
            for i, seg in enumerate(segments)
        ]

        out_file = dst / "transcript.json"
        out_file.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
        adapter.info(f"✅ Transcription saved in {time.time()-self.request.time_start:.2f}s")

        redis.publish(f"progress:{upload_id}", "50%"); redis.set(f"progress:{upload_id}", "50%")

    except Exception as e:
        adapter.error(f"❌ transcribe_segments failed: {e}", exc_info=True)
        redis.publish(f"progress:{upload_id}", "error"); redis.set(f"progress:{upload_id}", "error")
        raise

@shared_task(bind=True, name="tasks.diarize_full")
def diarize_full(self, upload_id: str, correlation_id: str):
    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    try:
        # WAV для диаризации
        folder = Path(settings.UPLOAD_FOLDER)
        wav = folder / f"{upload_id}.wav"
        if not wav.exists():
            # попытаемся сконвертировать из любого исходника
            srcs = list(folder.glob(f"{upload_id}.*"))
            if not srcs:
                raise FileNotFoundError(f"No source for {upload_id}")
            convert_to_wav(srcs[0], wav, sample_rate=16000, channels=1)

        adapter.info("Starting diarization")
        if settings.USE_FS_EEND:
            diar = get_clustering_diarizer()  # тут можно заменить на FS-EEND
        else:
            diar = get_clustering_diarizer()
        ann = diar.apply({"audio": str(wav)})

        speakers = [
            {"start": turn.start, "end": turn.end, "speaker": spk}
            for turn, _, spk in ann.itertracks(yield_label=True)
        ]

        dst = Path(settings.RESULTS_FOLDER) / upload_id
        dst.mkdir(exist_ok=True, parents=True)
        out_file = dst / "diarization.json"
        out_file.write_text(json.dumps(speakers, ensure_ascii=False, indent=2))
        adapter.info(f"✅ Diarization saved in {time.time()-self.request.time_start:.2f}s")

        redis.publish(f"progress:{upload_id}", "100%"); redis.set(f"progress:{upload_id}", "100%")

    except Exception as e:
        adapter.error(f"❌ diarize_full failed: {e}", exc_info=True)
        redis.publish(f"progress:{upload_id}", "error"); redis.set(f"progress:{upload_id}", "error")
        raise