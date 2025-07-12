import os
import json
import logging
import requests
import time
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
            logger.warning(f"Compute '{compute}' unsupported on CPU, using int8")
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
            getattr(settings, "VAD_MODEL_PATH", "pyannote/voice-activity-detection"),
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
    # прогрев моделей на старте процесса
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    device = settings.WHISPER_DEVICE.lower()
    if device == "cpu":
        try:
            opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
            get_whisper_model().transcribe(str(sample), **opts)
        except:
            pass
    else:
        try: get_vad().apply({"audio": str(sample)})
        except: pass
        try: get_clustering_diarizer().apply({"audio": str(sample)})
        except: pass

# === NO-OP для совместимости ===
@app.task(bind=True, name="tasks.download_audio", queue="preprocess_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")

# === Шаг 1: Preview (CPU) ===
@app.task(bind=True, name="tasks.preview_transcribe", queue="preprocess_cpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    upload_folder = Path(settings.UPLOAD_FOLDER)
    candidates = list(upload_folder.glob(f"{upload_id}.*"))
    if not candidates:
        logger.error(f"[{correlation_id}] Source for {upload_id} not found")
        return

    wav = upload_folder / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(candidates[0], wav)
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    model = get_whisper_model()
    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True, **opts)

    preview = {"text": "", "timestamps": []}
    for seg in segments:
        if seg.start >= settings.PREVIEW_LENGTH_S:
            break
        preview["text"] += seg.text
        preview["timestamps"].append({
            "start": seg.start, "end": seg.end, "text": seg.text
        })

    # записываем preview и прогресс
    redis_client.set(f"preview_result:{upload_id}", json.dumps(preview, ensure_ascii=False))
    state = {
        "status": "preview_done",
        "preview": preview,
        "chunks_total": 0,
        "chunks_done": 0,
        "diarize_requested": False
    }
    redis_client.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    redis_client.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    # колбэки
    for url in json.loads(redis_client.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(url, json={
                "external_id": upload_id,
                "event":       "preview_complete",
                "url_to_result": None
            }, timeout=5)
        except:
            pass

    # запускаем split → dispatch
    split_audio.delay(upload_id, correlation_id)

# === Шаг 2: Split audio на чанки (CPU) ===
@app.task(bind=True, name="tasks.split_audio", queue="preprocess_cpu")
def split_audio(self, upload_id: str, correlation_id: str):
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav_file = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav_file.exists():
        logger.error(f"[{correlation_id}] WAV for {upload_id} not found")
        return

    # загрузка и разбивка
    audio = AudioSegment.from_file(str(wav_file))
    chunk_length_ms = getattr(settings, "TRANSCRIPTION_CHUNK_LENGTH_S", 600) * 1000
    chunks_dir = Path(settings.UPLOAD_FOLDER) / upload_id / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = []
    for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start_ms:start_ms + chunk_length_ms]
        path = chunks_dir / f"chunk_{i}.wav"
        chunk.export(str(path), format="wav")
        chunk_files.append(str(path))

    dispatch_transcription.delay(upload_id, correlation_id, chunk_files)

# === Шаг 3: Dispatch GPU-транскрипции (CPU) ===
@app.task(bind=True, name="tasks.dispatch_transcription", queue="preprocess_cpu")
def dispatch_transcription(self, upload_id: str, correlation_id: str, chunk_files: list):
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    total = len(chunk_files)
    # прогресс после split
    preview = json.loads(redis_client.get(f"preview_result:{upload_id}") or "{}")
    state = {
        "status": "transcript_processing",
        "preview": preview,
        "chunks_total": total,
        "chunks_done": 0,
        "diarize_requested": bool(redis_client.get(f"diarize_requested:{upload_id}") == "1")
    }
    redis_client.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    redis_client.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    # запускаем GPU-задачи
    for path in chunk_files:
        transcribe_chunk.delay(upload_id, path, correlation_id)

# === Шаг 4: Транскрипция чанка (GPU) ===
@app.task(bind=True, name="tasks.transcribe_chunk", queue="preprocess_gpu")
def transcribe_chunk(self, upload_id: str, chunk_path: str, correlation_id: str):
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    model = get_whisper_model()
    opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
    segments, _ = model.transcribe(str(chunk_path), word_timestamps=True, **opts)

    # сохраняем части
    chunks_dir = Path(settings.RESULTS_FOLDER) / upload_id / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    idx = Path(chunk_path).stem.split("_")[1]
    out_file = chunks_dir / f"{idx}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([{"start": s.start, "end": s.end, "text": s.text} for s in segments], f, ensure_ascii=False, indent=2)

    # обновляем прогресс
    prev = redis_client.get(f"progress:{upload_id}")
    state = json.loads(prev) if prev else {}
    state["chunks_done"] = state.get("chunks_done", 0) + 1
    redis_client.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    redis_client.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    # когда все чанки готовы — собираем
    if state["chunks_done"] >= state["chunks_total"]:
        collect_transcription.delay(upload_id, correlation_id)

# === Шаг 5: Сборка полного транскрипта (GPU) ===
@app.task(bind=True, name="tasks.collect_transcription", queue="preprocess_gpu")
def collect_transcription(self, upload_id: str, correlation_id: str):
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    chunks_dir = Path(settings.RESULTS_FOLDER) / upload_id / "chunks"
    all_segs = []
    for j in sorted(chunks_dir.iterdir(), key=lambda p: int(p.stem)):
        all_segs.extend(json.loads(j.read_text(encoding="utf-8")))

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(
        json.dumps(all_segs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # финальный прогресс транскрипции
    prev = redis_client.get(f"progress:{upload_id}")
    state = json.loads(prev) if prev else {}
    state["status"] = "transcript_done"
    redis_client.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    redis_client.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    # колбэк
    for url in json.loads(redis_client.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(url, json={
                "external_id": upload_id,
                "event":       "transcript_complete"
            }, timeout=5)
        except:
            pass

    # по флагу запускаем диаризацию
    if redis_client.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)

# === Шаг 6: Диаризация (GPU, низкий приоритет) ===
@app.task(bind=True, name="tasks.diarize_full", queue="preprocess_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    state = json.loads(redis_client.get(f"progress:{upload_id}") or "{}")
    state["status"] = "diarization_processing"
    state["diarize_requested"] = True
    redis_client.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    redis_client.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    wav_file = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav_file.exists():
        logger.error(f"[{correlation_id}] WAV for {upload_id} not found")
        return

    annotation = get_clustering_diarizer().apply({"audio": str(wav_file)})
    segments = [{"start": float(s.start), "end": float(s.end), "speaker": sp}
                for s, _, sp in annotation.itertracks(yield_label=True)]

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diarization.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    state["status"] = "diarization_done"
    redis_client.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    redis_client.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    for url in json.loads(redis_client.get(f"callbacks:{upload_id}") or "[]"):
        try:
            requests.post(url, json={
                "external_id": upload_id,
                "event":       "diarization_complete"
            }, timeout=5)
        except:
            pass

# === Очистка старых файлов ===
@app.task(name="tasks.cleanup_old_uploads")
def cleanup_old_uploads():
    cutoff = time.time() - 24 * 3600
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)