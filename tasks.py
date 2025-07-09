# tasks.py

import os
import json
import logging
import time
from pathlib import Path

import torch
import requests
from celery import Celery, shared_task
from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from utils.audio import convert_to_wav
from pydub import AudioSegment
from redis import Redis

from config.settings import settings

# ─── Инициализация Celery ───────────────────────────────────────────────────────
app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
# По умолчанию — CPU-очередь
app.conf.task_default_queue = "preprocess_cpu"
# Маршрутизация: транскрипция→CPU, диаризация→GPU, внешняя транскрипция→external
app.conf.task_routes = {
    "tasks.transcribe_segments":    {"queue": "preprocess_cpu"},
    "tasks.diarize_full":            {"queue": "preprocess_gpu"},
  #  "tasks.external_transcribe":     {"queue": "preprocess_external"},
}

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device = settings.WHISPER_DEVICE.lower()
        compute = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(
                f"Compute type '{compute}' not supported on CPU; falling back to int8"
            )
            compute = "int8"
        logger.info(
            f"Loading WhisperModel (path={model_path}, device={device}, compute={compute})"
        )
        try:
            _whisper_model = WhisperModel(model_path, device=device, compute_type=compute)
        except Exception as e:
            logger.warning(
                f"Failed to load WhisperModel with compute={compute}: {e}; retrying int8"
            )
            _whisper_model = WhisperModel(
                model_path, device=device, compute_type="int8"
            )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_vad():
    global _vad
    if _vad is None:
        model_id = getattr(
            settings, "VAD_MODEL_PATH", "pyannote/voice-activity-detection"
        )
        _vad = VoiceActivityDetection.from_pretrained(
            model_id, use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("VAD pipeline loaded")
    return _vad


def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading clustering SpeakerDiarization into cache '{cache_dir}'")
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("Clustering-based diarizer loaded")
    return _clustering_diarizer


def get_fs_eend_pipeline():
    """
    Официальный end-to-end diarization pipeline из pyannote-audio.
    Идентификатор задаётся в settings.FS_EEND_PIPELINE.
    """
    pipeline_id = getattr(settings, "FS_EEND_PIPELINE", "pyannote/speaker-diarization")
    return SpeakerDiarization.from_pretrained(
        pipeline_id,
        use_auth_token=settings.HUGGINGFACE_TOKEN
    )


@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Для CPU-воркера (WHISPER_DEVICE=cpu) — прогреваем только Whisper.
    Для GPU-воркера (WHISPER_DEVICE=cuda) — прогреваем VAD + clustering (+ FS-EEND).
    """
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    device = settings.WHISPER_DEVICE.lower()

    if device == "cpu":
        try:
            opts = {}
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE
            get_whisper_model().transcribe(str(sample), **opts)
            logger.info("✅ Warm-up WhisperModel complete (CPU worker)")
        except Exception as e:
            logger.warning(f"Warm-up WhisperModel failed: {e}")
    else:
        try:
            get_vad().apply({"audio": str(sample)})
            logger.info("✅ Warm-up VAD complete (GPU worker)")
        except Exception as e:
            logger.warning(f"Warm-up VAD failed: {e}")

        try:
            get_clustering_diarizer().apply({"audio": str(sample)})
            logger.info("✅ Warm-up clustering diarizer complete (GPU worker)")
        except Exception as e:
            logger.warning(f"Warm-up clustering diarizer failed: {e}")

        if settings.USE_FS_EEND:
            try:
                pipe = get_fs_eend_pipeline()
                pipe({"audio": str(sample)})
                logger.info("✅ Warm-up FS-EEND complete (GPU worker)")
            except Exception as e:
                logger.warning(f"Warm-up FS-EEND failed: {e}")


@shared_task(bind=True, name="tasks.transcribe_segments", queue="preprocess_cpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    whisper = get_whisper_model()

    # ─── измеряем задержку в очереди ───────────────────────────────────────────────
    headers = getattr(self.request, "headers", {}) or {}
    enqueue_ts = headers.get("enqueue_time")
    if enqueue_ts:
        try:
            delay = time.time() - float(enqueue_ts)
            adapter.info(f"⏳ Queue delay: {delay:.2f}s")
        except ValueError:
            adapter.warning(f"Invalid enqueue_time header: {enqueue_ts}")

    start_time = time.time()
    upload_path = Path(settings.UPLOAD_FOLDER) / upload_id
    wav_name = Path(upload_id).stem + ".wav"
    wav_path = Path(settings.UPLOAD_FOLDER) / wav_name

    # ─── конвертация в WAV, если нужно ─────────────────────────────────────────────
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
    elapsed = time.time() - start_time
    adapter.info(f"✅ Transcription saved to {out_file} in {elapsed:.2f}s")

    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    redis.publish(f"progress:{upload_id}", "50%")
    redis.set(f"progress:{upload_id}", "50%")


@shared_task(bind=True, name="tasks.diarize_full", queue="preprocess_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})

    headers = getattr(self.request, "headers", {}) or {}
    enqueue_ts = headers.get("enqueue_time")
    if enqueue_ts:
        try:
            delay = time.time() - float(enqueue_ts)
            adapter.info(f"⏳ Queue delay: {delay:.2f}s")
        except ValueError:
            adapter.warning(f"Invalid enqueue_time header: {enqueue_ts}")

    start_time = time.time()
    wav_name = Path(upload_id).stem + ".wav"
    src = Path(settings.UPLOAD_FOLDER) / wav_name

    if not src.exists():
        orig = Path(settings.UPLOAD_FOLDER) / upload_id
        if orig.exists():
            adapter.info(f"Converting '{orig}' to WAV for diarization")
            convert_to_wav(orig, src, sample_rate=16000, channels=1)
        else:
            raise ValueError(f"Neither {src} nor {orig} exist for upload_id={upload_id}")

    speech = get_vad().apply({"audio": str(src)})

    speakers = []
    if settings.USE_FS_EEND:
        adapter.info("Using FS-EEND (pyannote) diarization")
        pipeline = get_fs_eend_pipeline()
        diarization = pipeline({"audio": str(src)})
        for segment, _, spk in diarization.itertracks(yield_label=True):
            speakers.append({"start": segment.start, "end": segment.end, "speaker": spk})
    else:
        adapter.info("Using clustering-based diarization")
        ann = get_clustering_diarizer().apply({"audio": str(src)})
        for turn, _, spk in ann.itertracks(yield_label=True):
            speakers.append({"start": turn.start, "end": turn.end, "speaker": spk})

    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_file = dst_dir / "diarization.json"
    out_file.write_text(json.dumps(speakers, ensure_ascii=False, indent=2))
    elapsed = time.time() - start_time
    adapter.info(f"✅ Diarization saved to {out_file} in {elapsed:.2f}s")

    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    redis.publish(f"progress:{upload_id}", "100%")
    redis.set(f"progress:{upload_id}", "100%")


@shared_task(bind=True, name="tasks.external_transcribe", queue="preprocess_external")
def external_transcribe(self, upload_id: str, correlation_id: str):
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    file_url = f"{settings.PUBLIC_BASE_URL}/uploads/{upload_id}"
    headers = {"API-KEY": settings.EXTERNAL_API_KEY}
    payload = {"upload": file_url}

    try:
        response = requests.post(f"{settings.EXTERNAL_API_URL}/upload", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        ext_id = data.get("id")
        if not ext_id:
            raise ValueError("No 'id' returned by external API")
        adapter.info(f"Started external transcription, id={ext_id}")
    except Exception as e:
        adapter.error(f"Failed to start external transcription: {e}", exc_info=True)
        return

    # Poll until completion
    status_url = f"{settings.EXTERNAL_API_URL}/status/{ext_id}"
    while True:
        try:
            status_resp = requests.get(status_url, headers=headers)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status")
            adapter.info(f"External status: {status}")
            if status == "completed":
                result_url = status_data.get("result_url")
                break
            if status == "error":
                adapter.error(f"External transcription error: {status_data}")
                return
        except Exception as e:
            adapter.warning(f"Polling error: {e}", exc_info=True)
        time.sleep(settings.EXTERNAL_POLL_INTERVAL_S)

    # Fetch and save result
    try:
        result_resp = requests.get(result_url)
        result_resp.raise_for_status()
        transcript = result_resp.json()
    except Exception as e:
        adapter.error(f"Failed to fetch external transcript: {e}", exc_info=True)
        return

    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_file = dst_dir / "transcript.json"
    out_file.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    adapter.info(f"Saved external transcript to {out_file}")

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