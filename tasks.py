import os
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from celery.signals import worker_process_init
from redis import Redis

from config.settings import settings
from config.celery import celery_app

logger = logging.getLogger(__name__)

# faster-whisper
try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info(f"[{datetime.utcnow().isoformat()}] [INIT] faster-whisper available")
except ImportError as e:
    _HF_AVAILABLE = False
    logger.warning(f"[{datetime.utcnow().isoformat()}] [INIT] faster-whisper not available: {e}")

# pyannote.audio
try:
    from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
    _PN_AVAILABLE = True
    logger.info(f"[{datetime.utcnow().isoformat()}] [INIT] pyannote.audio available")
except ImportError as e:
    _PN_AVAILABLE = False
    logger.warning(f"[{datetime.utcnow().isoformat()}] [INIT] pyannote.audio not available: {e}")

_whisper_model = None
_vad = None
_clustering_diarizer = None


def get_whisper_model(model_override: str = None):
    global _whisper_model
    if model_override:
        # возможность смены модели для preview
        model_id = model_override
        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "float16")
        device = settings.WHISPER_DEVICE.lower()
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] override model {model_id}")
        return WhisperModel(model_override, device=device, compute_type=compute)
    if _whisper_model is None:
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] initializing model")
        model_id = settings.WHISPER_MODEL_PATH
        cache    = settings.HUGGINGFACE_CACHE_DIR
        device   = settings.WHISPER_DEVICE.lower()
        local    = (device == "cpu")
        try:
            path = download_model(model_id, cache_dir=cache, local_files_only=local)
        except Exception:
            path = model_id
        compute = getattr(
            settings,
            "WHISPER_COMPUTE_TYPE",
            "float16" if device.startswith("cuda") else "int8"
        ).lower()
        if device == "cpu" and compute in ("fp16", "float16"):
            compute = "int8"
        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info(f"[{datetime.utcnow().isoformat()}] [WHISPER] model ready on {device} ({compute})")
    return _whisper_model


def get_vad():
    global _vad
    if _vad is None:
        logger.info(f"[{datetime.utcnow().isoformat()}] [VAD] loading VAD")
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
        logger.info(f"[{datetime.utcnow().isoformat()}] [DIARIZER] loading diarizer")
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
    logger.info(f"[{datetime.utcnow().isoformat()}] [WARMUP] worker init — warming up models")
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
    if _PN_AVAILABLE and settings.WHISPER_DEVICE.lower().startswith("cuda"):
        try:
            get_vad(); get_clustering_diarizer()
            logger.info(f"[{datetime.utcnow().isoformat()}] [WARMUP] VAD & diarizer warmup ok")
        except Exception:
            logger.warning(f"[{datetime.utcnow().isoformat()}] [WARMUP] VAD/diarizer warmup failed")


def prepare_wav(upload_id: str) -> Path:
    """
    Забираем любой файл upload_id.* из каталога загрузок и
    конвертируем (или переименовываем) его в UPLOAD_FOLDER/<upload_id>.wav.
    """
    logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] start prepare_wav for {upload_id}")
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "stream=codec_name,sample_rate,channels",
             "-of", "default=noprint_wrappers=1",
             str(src)],
            capture_output=True, text=True, check=True
        )
        info = {l.split("=")[0]: l.split("=")[1] for l in probe.stdout.splitlines()}
        if (src.suffix.lower() == ".wav"
            and info.get("codec_name") == "pcm_s16le"
            and info.get("sample_rate") == "16000"
            and info.get("channels") == "1"):
            if src != target:
                src.rename(target)
            logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] WAV ready (renamed) {target.name}")
            return target
    except Exception:
        pass

    # конвертация
    threads = getattr(settings, "FFMPEG_THREADS", 2)
    subprocess.run([
        "ffmpeg", "-y", "-threads", str(threads),
        "-i", str(src),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        str(target)
    ], check=True)
    logger.info(f"[{datetime.utcnow().isoformat()}] [PREPARE] WAV converted to {target.name}")
    return target


@celery_app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] CONVERT task received for {upload_id}")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    try:
        wav = prepare_wav(upload_id)
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] convert_to_wav failed", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        return

    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] WAV ready, calling preview_transcribe")
    from tasks import preview_transcribe
    preview_transcribe.delay(upload_id, correlation_id)
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] CONVERT done")


@celery_app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    """
    Быстрый PREVIEW: один вызов Whisper с max_initial_timestamp,
    без FFmpeg-пайпов, напрямую по файлу upload_id.wav.
    """
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] PREVIEW task received for {upload_id}")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav.exists():
        err = "preview source WAV not found"
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] {err}")
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":err}))
        return

    try:
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] PREVIEW calling WhisperModel.transcribe")
        # выбор модели для preview (опционально)
        model = (get_whisper_model(settings.PREVIEW_WHISPER_MODEL)
                 if getattr(settings, "PREVIEW_WHISPER_MODEL", None)
                 else get_whisper_model())

        segments, _ = model.transcribe(
            str(wav),
            word_timestamps=True,
            max_initial_timestamp=settings.PREVIEW_LENGTH_S,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] PREVIEW Whisper returned {len(segments)} segments")
        for seg in segments:
            r.publish(
                f"progress:{upload_id}",
                json.dumps({
                    "status":        "preview_partial",
                    "fragment":      {"start": seg.start, "end": seg.end, "text": seg.text}
                })
            )
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] preview error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        return

    preview = {
        "text":       "".join(s.text for s in segments),
        "timestamps": [{"start":s.start, "end":s.end, "text":s.text} for s in segments]
    }
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview_transcript.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2)
    )
    r.publish(f"progress:{upload_id}", json.dumps({"status":"preview_done","preview":preview}))
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] PREVIEW done, enqueued full transcription")

    from tasks import transcribe_segments
    transcribe_segments.delay(upload_id, correlation_id)


@celery_app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] TRANSCRIBE task received for {upload_id}")
    if not _HF_AVAILABLE:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] no Whisper available, skipping")
        return
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not wav.exists():
        err = "transcribe source WAV not found"
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] {err}")
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":err}))
        return

    try:
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] TRANSCRIBE calling WhisperModel.transcribe")
        segments, _ = get_whisper_model().transcribe(
            str(wav),
            word_timestamps=True,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {})
        )
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] TRANSCRIBE Whisper returned {len(segments)} segments")
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] transcribe error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        return

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(
        json.dumps([{"start":s.start,"end":s.end,"text":s.text} for s in segments],
                   ensure_ascii=False, indent=2)
    )
    r.publish(f"progress:{upload_id}", json.dumps({"status":"transcript_done"}))
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] TRANSCRIBE done")


@celery_app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    cid = correlation_id or "?"
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] DIARIZE task received for {upload_id}")
    if not _PN_AVAILABLE:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] pyannote not available, skipping")
        return
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status":"diarize_started"}))

    try:
        wav = prepare_wav(upload_id)
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] DIARIZE calling VAD and diarizer")
        speech = get_vad().apply({"audio":str(wav)})
        ann    = get_clustering_diarizer().apply({"audio":str(wav),"speech":speech})
        segs   = [{"start":float(s.start),"end":float(s.end),"speaker":spk}
                  for s,_,spk in ann.itertracks(yield_label=True)]
        logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] DIARIZE ready {len(segs)} segments")
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] [{cid}] diarize error", exc_info=True)
        r.publish(f"progress:{upload_id}", json.dumps({"status":"error","error":str(e)}))
        return

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diarization.json").write_text(json.dumps(segs, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status":"diarization_done"}))
    logger.info(f"[{datetime.utcnow().isoformat()}] [{cid}] DIARIZE done")


@celery_app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    age = settings.FILE_RETENTION_DAYS
    cutoff = datetime.utcnow() - timedelta(days=age)
    deleted = 0
    for base in (Path(settings.UPLOAD_FOLDER), Path(settings.RESULTS_FOLDER)):
        for p in base.glob("**/*"):
            try:
                file_age = datetime.utcnow() - datetime.fromtimestamp(p.stat().st_mtime)
                if file_age > timedelta(days=age):
                    if p.is_dir():
                        p.rmdir()
                    else:
                        p.unlink()
                    deleted += 1
            except Exception:
                continue
    logger.info(f"[{datetime.utcnow().isoformat()}] [CLEANUP] deleted {deleted} old files")