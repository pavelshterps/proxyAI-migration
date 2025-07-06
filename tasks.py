import os
import json
import logging
import time
from pathlib import Path

import torch
from celery import Celery, shared_task
from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from utils.audio import convert_to_wav
from pydub import AudioSegment
from redis import Redis

# FS-EEND import
from eend.inference import Inference as EENDInference

from config.settings import settings

# Инициализация Celery
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
app.conf.task_default_queue = "preprocess_cpu"

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_diarizer = None
_eend_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device = settings.WHISPER_DEVICE.lower()
        compute = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(f"Compute type '{compute}' not supported on CPU; falling back to int8")
            compute = "int8"
        logger.info(f"Loading WhisperModel (path={model_path}, device={device}, compute={compute})")
        try:
            _whisper_model = WhisperModel(model_path, device=device, compute_type=compute)
        except Exception as e:
            logger.warning(f"Failed to load WhisperModel with compute={compute}: {e}; retrying int8")
            _whisper_model = WhisperModel(model_path, device=device, compute_type="int8")
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_vad():
    global _vad
    if _vad is None:
        model_id = getattr(settings, "VAD_MODEL_PATH", "pyannote/voice-activity-detection")
        _vad = VoiceActivityDetection.from_pretrained(
            model_id, use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        # Переносим на GPU, если нужно
        device = settings.FS_EEND_DEVICE.lower() if settings.USE_FS_EEND else settings.WHISPER_DEVICE.lower()
        if device != "cpu":
            try:
                _vad.to(torch.device(device))
                logger.info(f"VAD pipeline moved to {device}")
            except Exception as e:
                logger.warning(f"Could not move VAD to {device}: {e}")
        logger.info("VAD pipeline loaded")
    return _vad


def get_clustering_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading clustering SpeakerDiarization into cache '{cache_dir}'")
        _diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        device = getattr(settings, "PYANNOTE_DEVICE", settings.WHISPER_DEVICE).lower()
        if device != "cpu":
            try:
                _diarizer.to(torch.device(device))
                logger.info(f"Clustering diarizer moved to {device}")
            except Exception as e:
                logger.warning(f"Could not move clustering diarizer to {device}: {e}")
        logger.info("Clustering-based diarizer loaded")
    return _diarizer


def get_eend_model():
    global _eend_model
    if _eend_model is None:
        model_path = settings.FS_EEND_MODEL_PATH
        device = settings.FS_EEND_DEVICE.lower()
        _eend_model = EENDInference(model=model_path, device=device)
        logger.info(f"FS-EEND model loaded (path={model_path}, device={device})")
    return _eend_model


@worker_process_init.connect
def preload_and_warmup(**kwargs):
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    # Warm-up VAD
    try:
        get_vad().apply({"audio": str(sample)})
        logger.info("✅ Warm-up VAD complete")
    except Exception as e:
        logger.warning(f"Warm-up VAD failed: {e}")
    # Warm-up clustering diarizer
    try:
        get_clustering_diarizer().apply({"audio": str(sample)})
        logger.info("✅ Warm-up clustering diarizer complete")
    except Exception as e:
        logger.warning(f"Warm-up clustering diarizer failed: {e}")
    # Warm-up FS-EEND
    if settings.USE_FS_EEND and settings.FS_EEND_MODEL_PATH:
        try:
            get_eend_model().diarize(str(sample))
            logger.info("✅ Warm-up FS-EEND complete")
        except Exception as e:
            logger.warning(f"Warm-up FS-EEND failed: {e}")
    # Warm-up Whisper
    try:
        opts = {}
        if settings.WHISPER_LANGUAGE:
            opts["language"] = settings.WHISPER_LANGUAGE
        get_whisper_model().transcribe(str(sample), **opts)
        logger.info("✅ Warm-up WhisperModel complete")
    except Exception as e:
        logger.warning(f"Warm-up WhisperModel failed: {e}")


@shared_task(bind=True, name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    whisper = get_whisper_model()

    # Измеряем задержку в очереди
    headers = getattr(self.request, "headers", {}) or {}
    enqueue_ts = headers.get("enqueue_time")
    if enqueue_ts:
        try:
            delay = time.time() - float(enqueue_ts)
            adapter.info(f"⏳ Queue delay: {delay:.2f}s")
        except ValueError:
            adapter.warning(f"Invalid enqueue_time header: {enqueue_ts}")

    start_time = time.time()

    # Конвертация в WAV при необходимости
    upload_path = Path(settings.UPLOAD_FOLDER) / upload_id
    wav_name = Path(upload_id).stem + ".wav"
    wav_path = Path(settings.UPLOAD_FOLDER) / wav_name
    if upload_path.suffix.lower() != ".wav":
        convert_to_wav(upload_path, wav_path, sample_rate=16000, channels=1)
        src = wav_path
    else:
        src = upload_path

    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    adapter.info(f"Starting transcription for '{src}'")
    audio = AudioSegment.from_file(str(src))
    transcript = []
    windows = split_audio_fixed_windows(src, settings.SEGMENT_LENGTH_S)

    for idx, (start, end) in enumerate(windows):
        adapter.debug(f"  Segment {idx}: {start:.1f}-{end:.1f}s")
        chunk = audio[int(start * 1000):int(end * 1000)]
        tmp_wav = dst_dir / f"{upload_id}_{idx}.wav"
        chunk.export(tmp_wav, format="wav")

        try:
            opts = {"beam_size": settings.WHISPER_BATCH_SIZE, "word_timestamps": True}
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE
            segments, _ = whisper.transcribe(str(tmp_wav), **opts)
        except Exception as e:
            adapter.error(f"❌ Transcription failed on segment {idx}: {e}", exc_info=True)
            tmp_wav.unlink()
            continue

        for seg in segments:
            transcript.append({
                "segment": idx,
                "start": start + seg.start,
                "end": start + seg.end,
                "text": seg.text,
            })
        tmp_wav.unlink()

    out_file = dst_dir / "transcript.json"
    out_file.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    adapter.info(f"Transcription saved to {out_file} in {time.time() - start_time:.2f}s")

    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    redis.publish(f"progress:{upload_id}", "50%")
    redis.set(f"progress:{upload_id}", "50%")


@shared_task(bind=True, name="tasks.diarize_full", queue="preprocess_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    start_time = time.time()

    # Путь к WAV
    wav_name = Path(upload_id).stem + ".wav"
    src = Path(settings.UPLOAD_FOLDER) / wav_name

    # VAD-предобработка
    vad = get_vad()
    speech = vad.apply({"audio": str(src)})
    regions = speech.get_timeline().support()

    speakers = []
    if settings.USE_FS_EEND and settings.FS_EEND_MODEL_PATH:
        adapter.info("Using FS-EEND diarization")
        eend = get_eend_model()
        labels = eend.diarize(str(src))
        fs = settings.FRAME_SHIFT
        for i, row in enumerate(labels):
            t0 = i * fs
            for spk_idx, active in enumerate(row):
                if active:
                    speakers.append({"start": t0, "end": t0 + fs, "speaker": spk_idx})
    else:
        adapter.info("Using clustering-based diarization")
        diar = get_clustering_diarizer()
        ann = diar.apply({"audio": str(src)})
        for turn, _, spk in ann.itertracks(yield_label=True):
            speakers.append({"start": turn.start, "end": turn.end, "speaker": spk})

    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_file = dst_dir / "diarization.json"
    out_file.write_text(json.dumps(speakers, ensure_ascii=False, indent=2))
    adapter.info(f"Diarization saved to {out_file} in {time.time() - start_time:.2f}s")

    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    redis.publish(f"progress:{upload_id}", "100%")
    redis.set(f"progress:{upload_id}", "100%")

def split_audio_fixed_windows(audio_path: Path, window_s: int):
    audio = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    return [
        (start / 1000.0, min(start + window_ms, length_ms) / 1000.0)
        for start in range(0, length_ms, window_ms)
    ]