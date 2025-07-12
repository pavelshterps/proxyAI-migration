import os
import json
import logging
import time
import requests
from pathlib import Path

from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from redis import Redis

from config.settings import settings
from config.celery import app          # единый Celery instance
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device     = settings.WHISPER_DEVICE.lower()
        compute    = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(
                f"Compute type '{compute}' not supported on CPU; falling back to int8"
            )
            compute = "int8"
        _whisper_model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute
        )
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        model_id = getattr(
            settings,
            "VAD_MODEL_PATH",
            "pyannote/voice-activity-detection"
        )
        _vad = VoiceActivityDetection.from_pretrained(
            model_id, use_auth_token=settings.HUGGINGFACE_TOKEN
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
    """
    Прогрев моделей один раз при старте воркера.
    """
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    device = settings.WHISPER_DEVICE.lower()
    if device == "cpu":
        try:
            opts = {}
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE
            get_whisper_model().transcribe(str(sample), **opts)
        except:
            pass
    else:
        try:
            get_vad().apply({"audio": str(sample)})
        except:
            pass
        try:
            get_clustering_diarizer().apply({"audio": str(sample)})
        except:
            pass

@app.task(bind=True, name="tasks.download_audio", queue="preprocess_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")

@app.task(bind=True, name="tasks.preview_transcribe", queue="preprocess_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    """Preview: первые settings.PREVIEW_LENGTH_S секунд."""
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    # 1) Конвертация исходного файла в WAV
    upload_folder = Path(settings.UPLOAD_FOLDER)
    candidates = list(upload_folder.glob(f"{upload_id}.*"))
    if not candidates:
        logger.error(f"[{correlation_id}] Source file for {upload_id} not found")
        return
    src = candidates[0]
    dst = upload_folder / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(src, dst)
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    # 2) Транскрипция первых N секунд
    model = get_whisper_model()
    opts = {}
    if settings.WHISPER_LANGUAGE:
        opts["language"] = settings.WHISPER_LANGUAGE
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True, **opts)

    preview = {"text": "", "timestamps": []}
    for seg in segments:
        if seg.start >= settings.PREVIEW_LENGTH_S:
            break
        preview["text"] += seg.text
        preview["timestamps"].append({
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text
        })

    # 3) Сохраняем preview и обновляем прогресс
    redis_client.set(f"preview_result:{upload_id}", json.dumps(preview, ensure_ascii=False))
    redis_client.set(f"progress:{upload_id}", "preview_done")
    redis_client.publish(f"progress:{upload_id}", "preview_done")

    # 4) callbacks preview_complete
    for url in json.loads(redis_client.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(
                url,
                json={
                    "external_id":   upload_id,
                    "event":         "preview_complete",
                    "url_to_result": None
                },
                timeout=5
            )
        except:
            pass

    # 5) Запускаем полный транскрипт
    transcribe_segments.delay(upload_id, correlation_id)

@app.task(bind=True, name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    """Полная транскрипция."""
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    # 1) Конвертация в WAV
    upload_folder = Path(settings.UPLOAD_FOLDER)
    candidates = list(upload_folder.glob(f"{upload_id}.*"))
    if not candidates:
        logger.error(f"[{correlation_id}] Source file for {upload_id} not found")
        return
    src = candidates[0]
    dst = upload_folder / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(src, dst)
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    # 2) Полная транскрипция
    model = get_whisper_model()
    opts = {}
    if settings.WHISPER_LANGUAGE:
        opts["language"] = settings.WHISPER_LANGUAGE
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True, **opts)

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript = [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 3) Обновляем прогресс и callbacks
    redis_client.set(f"progress:{upload_id}", "transcript_done")
    redis_client.publish(f"progress:{upload_id}", "transcript_done")
    for url in json.loads(redis_client.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(
                url,
                json={"external_id": upload_id, "event": "transcript_complete"},
                timeout=5
            )
        except:
            pass

    # 4) Если запросили — сразу в очередь на диаризацию
    if redis_client.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

@app.task(bind=True, name="tasks.diarize_full", queue="preprocess_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    """Полная диаризация."""
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    wav_file = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav_file.exists():
        logger.error(f"[{correlation_id}] WAV for {upload_id} not found")
        return

    # 1) Диаризация (VAD+clustering внутри пайплайна)
    annotation = get_clustering_diarizer().apply({"audio": str(wav_file)})

    # 2) Собираем сегменты
    segments = []
    for segment, track, label in annotation.itertracks(yield_label=True):
        segments.append({
            "start": float(segment.start),
            "end":   float(segment.end),
            "speaker": label
        })

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diarization.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 3) Обновляем прогресс и callbacks
    redis_client.set(f"progress:{upload_id}", "diarization_done")
    redis_client.publish(f"progress:{upload_id}", "diarization_done")
    for url in json.loads(redis_client.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(
                url,
                json={"external_id": upload_id, "event": "diarization_complete"},
                timeout=5
            )
        except:
            pass

@app.task(name="tasks.cleanup_old_uploads")
def cleanup_old_uploads():
    """Удаление старых файлов старше суток."""
    cutoff = time.time() - 24 * 3600
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)