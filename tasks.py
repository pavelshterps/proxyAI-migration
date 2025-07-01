import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import structlog
import webrtcvad
from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
from prometheus_client import Counter, Summary

from config.settings import settings

# structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# Metrics
TASK_RUNS = Counter("celery_task_runs_total", "Celery task runs", ["task"])
VAD_TIME = Summary("vad_segmentation_seconds", "VAD segmentation duration")
TRANSCRIBE_TIME = Summary("whisper_transcription_seconds", "Whisper transcription duration")

# Singletons
_whisper = None
_diarizer = None

def get_whisper():
    global _whisper
    if _whisper is None:
        logger.info("loading whisper model", path=settings.WHISPER_MODEL_PATH)
        _whisper = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
            batch_size = settings.BATCH_SIZE,
            cache_dir = settings.DIARIZER_CACHE_DIR,
            local_files_only = True
        )
        logger.info("whisper model loaded")
    return _whisper

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache = Path(settings.DIARIZER_CACHE_DIR)
        cache.mkdir(parents=True, exist_ok=True)
        logger.info("loading diarizer model", protocol=settings.PYANNOTE_PROTOCOL)
        try:
            _diarizer = Pipeline.from_pretrained(
                settings.PYANNOTE_PROTOCOL,
                cache_dir=str(cache),
                use_auth_token=settings.HUGGINGFACE_TOKEN
            )
            logger.info("diarizer model loaded")
        except Exception as e:
            logger.error("diarizer loading failed", error=str(e))
            raise
    return _diarizer

def vad_segments(audio_path: Path):
    start = perf_counter()
    try:
        audio = AudioSegment.from_file(str(audio_path)).set_channels(1).set_frame_rate(16000)
        raw = audio.raw_data
        vad = webrtcvad.Vad(settings.VAD_LEVEL)
        frame_ms = 30
        bytes_per_frame = int(16000 * frame_ms / 1000 * 2)
        segments, current, ts = [], None, 0.0
        for i in range(0, len(raw), bytes_per_frame):
            frame = raw[i:i+bytes_per_frame]
            is_speech = vad.is_speech(frame, sample_rate=16000)
            if is_speech and current is None:
                current = ts
            if not is_speech and current is not None:
                segments.append((current, ts))
                current = None
            ts += frame_ms / 1000
        if current is not None:
            segments.append((current, ts))
        if not segments:
            raise RuntimeError("no speech detected")
        return segments
    except Exception as e:
        logger.warning("VAD failed, using fixed windows", error=str(e))
        length = AudioSegment.from_file(str(audio_path)).duration_seconds
        seg = settings.SEGMENT_LENGTH_S
        return [(i, min(i+seg, length)) for i in range(0, int(length), seg)]
    finally:
        VAD_TIME.observe(perf_counter() - start)

@shared_task(name="tasks.transcribe_segments", bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def transcribe_segments(self, upload_id: str, correlation_id: str | None = None):
    TASK_RUNS.labels(task="transcribe_segments").inc()
    whisper = get_whisper()
    src = Path(settings.UPLOAD_FOLDER) / upload_id
    dst = Path(settings.RESULTS_FOLDER) / upload_id
    dst.mkdir(exist_ok=True, parents=True)
    logger_ctx = logger.bind(upload_id=upload_id, correlation_id=correlation_id)
    logger_ctx.info("transcription start")
    segments = vad_segments(src)
    out = []
    for idx, (start, end) in enumerate(segments):
        with TRANSCRIBE_TIME.time():
            try:
                r = whisper.transcribe(
                    str(src),
                    offset=start,
                    duration=end-start,
                    language="ru",
                    vad_filter=True,
                    word_timestamps=True
                )
                text = " ".join(s["text"] for s in r["segments"])
                out.append({"segment": idx, "start": start, "end": end, "text": text})
                logger_ctx.debug("segment done", idx=idx)
            except Exception as e:
                logger_ctx.error("segment error", idx=idx, error=str(e))
                out.append({"segment": idx, "start": start, "end": end, "text": "", "error": str(e)})
    (dst / "transcript.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger_ctx.info("transcription complete")

@shared_task(name="tasks.diarize_full", bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2})
def diarize_full(self, upload_id: str, correlation_id: str | None = None):
    TASK_RUNS.labels(task="diarize_full").inc()
    diarizer = get_diarizer()
    src = Path(settings.UPLOAD_FOLDER) / upload_id
    dst = Path(settings.RESULTS_FOLDER) / upload_id
    dst.mkdir(exist_ok=True, parents=True)
    logger_ctx = logger.bind(upload_id=upload_id, correlation_id=correlation_id)
    logger_ctx.info("diarization start")
    try:
        res = diarizer(str(src))
        segs = [{"start": t.start, "end": t.end, "speaker": s} for t, _, s in res.itertracks(yield_label=True)]
    except Exception as e:
        logger_ctx.error("diarization error", error=str(e))
        segs = []
    (dst / "diarization.json").write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    logger_ctx.info("diarization complete")

@shared_task(name="tasks.cleanup_old_files")
def cleanup_old_files():
    TASK_RUNS.labels(task="cleanup_old_files").inc()
    cutoff = datetime.utcnow() - timedelta(days=settings.FILE_RETENTION_DAYS)
    for f in Path(settings.UPLOAD_FOLDER).iterdir():
        if f.is_file() and datetime.utcfromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink(missing_ok=True)
    for d in Path(settings.RESULTS_FOLDER).iterdir():
        if d.is_dir() and datetime.utcfromtimestamp(d.stat().st_mtime) < cutoff:
            shutil.rmtree(d, ignore_errors=True)
    logger.info("cleanup complete", retention=settings.FILE_RETENTION_DAYS)