import json
import logging
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from celery.signals import worker_process_init
from redis import Redis

from config.settings import settings
from config.celery import celery_app

logger = logging.getLogger(__name__)

# Импорт Whisper
try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info("[INIT] faster_whisper available")
except ImportError as e:
    _HF_AVAILABLE = False
    logger.warning(f"[INIT] faster_whisper not available: {e}")

# Импорт Pyannote
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

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("[WHISPER] initializing model")
        model_id = settings.WHISPER_MODEL_PATH
        cache    = settings.HUGGINGFACE_CACHE_DIR
        device   = settings.WHISPER_DEVICE.lower()
        local    = (device == "cpu")
        try:
            path = download_model(model_id, cache_dir=cache, local_files_only=local)
        except:
            path = model_id
        compute = getattr(
            settings,
            "WHISPER_COMPUTE_TYPE",
            ("float16" if device.startswith("cuda") else "int8")
        ).lower()
        if device == "cpu" and compute in ("fp16", "float16"):
            compute = "int8"
        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info(f"[WHISPER] model ready on {device} ({compute})")
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        logger.info("[VAD] loading VAD")
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("[VAD] ready")
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        logger.info("[DIARIZER] loading diarizer")
        Path(settings.DIARIZER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("[DIARIZER] ready")
    return _clustering_diarizer

@worker_process_init.connect
def preload_on_startup(**kwargs):
    logger.info("[WARMUP] worker init — warming up models")
    if _HF_AVAILABLE:
        sample = Path(__file__).parent / "tests/fixtures/sample.wav"
        try:
            get_whisper_model().transcribe(
                str(sample),
                max_initial_timestamp=settings.PREVIEW_LENGTH_S
            )
            logger.info("[WARMUP] Whisper warmup ok")
        except:
            logger.warning("[WARMUP] Whisper warmup failed")
    if _PN_AVAILABLE and settings.WHISPER_DEVICE.lower().startswith("cuda"):
        try:
            get_vad()
            get_clustering_diarizer()
            logger.info("[WARMUP] VAD & diarizer warmup ok")
        except:
            logger.warning("[WARMUP] VAD/diarizer warmup failed")

# Preview + full transcription on GPU queue
@celery_app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{cid}] PREVIEW TRANSCRIBE start {upload_id}")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    try:
        # 1) Обрезка превью
        src     = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(exist_ok=True, parents=True)
        preview_wav = out_dir / f"{upload_id}_preview.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ss", "0", "-t", str(settings.PREVIEW_LENGTH_S),
             "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16k",
             str(preview_wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # 2) Транскрипция превью и публикация частичных результатов
        segments, info = get_whisper_model().transcribe(
            str(preview_wav),
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        segment_acc = []
        for seg in segments:
            frag = {"start": seg.start, "end": seg.end, "text": seg.text}
            segment_acc.append(frag)
            r.publish(
                f"progress:{upload_id}",
                json.dumps({"status": "preview_partial", "fragment": frag})
            )
        segs = segment_acc

    except Exception as e:
        logger.error(f"[{cid}] preview_transcribe error", exc_info=True)
        r.publish(
            f"progress:{upload_id}",
            json.dumps({"status": "error", "error": str(e)})
        )
        return

    # Финальный результат превью
    preview = {
        "text": "".join(s["text"] for s in segs),
        "timestamps": segs
    }
    (out_dir / "preview_transcript.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2)
    )
    r.publish(
        f"progress:{upload_id}",
        json.dumps({"status": "preview_done", "preview": preview})
    )

    # Запуск полного транскрипта
    transcribe_segments.delay(upload_id, correlation_id)
    logger.info(f"[{cid}] PREVIEW TRANSCRIBE done in {time.time()-t0:.2f}s")

@celery_app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{cid}] TRANSCRIBE start {upload_id}")
    if not _HF_AVAILABLE:
        logger.error(f"[{cid}] no Whisper, skip transcribe")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        segs, _ = get_whisper_model().transcribe(
            str(wav),
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
    except Exception as e:
        logger.error(f"[{cid}] transcribe_segments error", exc_info=True)
        r.publish(
            f"progress:{upload_id}",
            json.dumps({"status": "error", "error": str(e)})
        )
        return

    segs = list(segs)
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(exist_ok=True)
    (out / "transcript.json").write_text(
        json.dumps(
            [{"start": s.start, "end": s.end, "text": s.text} for s in segs],
            ensure_ascii=False,
            indent=2
        )
    )
    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))

    if r.get(f"diarize_requested:{upload_id}") == "1":
        diarize_full.delay(upload_id, correlation_id)
        logger.info(f"[{cid}] auto-diarize queued")

    logger.info(f"[{cid}] TRANSCRIBE done in {time.time()-t0:.2f}s")

@celery_app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{cid}] DIARIZE start {upload_id}")
    if not _PN_AVAILABLE:
        logger.error(f"[{cid}] no pyannote, skip diarize")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        get_vad()
        get_clustering_diarizer()
        ann = get_clustering_diarizer().apply({"audio": str(wav)})
    except Exception as e:
        logger.error(f"[{cid}] diarize_full error", exc_info=True)
        r.publish(
            f"progress:{upload_id}",
            json.dumps({"status": "error", "error": str(e)})
        )
        return

    segs = [
        {"start": float(s.start), "end": float(s.end), "speaker": spk}
        for s, _, spk in ann.itertracks(yield_label=True)
    ]
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(exist_ok=True)
    (out / "diarization.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=2)
    )
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarization_done"}))
    logger.info(f"[{cid}] DIARIZE done in {time.time()-t0:.2f}s")

@celery_app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    """
    Удаляет файлы старше settings.FILE_RETENTION_DAYS.
    """
    age_limit = settings.FILE_RETENTION_DAYS
    cutoff = datetime.utcnow() - timedelta(days=age_limit)
    deleted = 0
    for base in (Path(settings.UPLOAD_FOLDER), Path(settings.RESULTS_FOLDER)):
        for path in base.glob("**/*"):
            try:
                if datetime.utcfromtimestamp(path.stat().st_mtime) < cutoff:
                    (path.rmdir() if path.is_dir() else path.unlink())
                    deleted += 1
            except:
                continue
    logger.info(f"[CLEANUP] удалено {deleted} устаревших файлов/папок")