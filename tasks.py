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
from pyannote.audio.exceptions import ModelNotFoundError
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

# Метрики
TASK_RUN_COUNTER = Counter("proxyai_task_runs_total", "Total Celery task runs", ["task_name"])
SEGMENTATION_DURATION = Summary("proxyai_segmentation_seconds", "Time spent in VAD segmentation")
TRANSCRIPTION_DURATION = Summary("proxyai_transcription_seconds", "Time spent in Whisper transcription")

# Singletons
_whisper_model = None
_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            logger.info(
                "loading_whisper_model",
                path=settings.whisper_model_path,
                device=settings.whisper_device,
                compute=settings.whisper_compute_type
            )
            _whisper_model = WhisperModel(
                settings.whisper_model_path,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type
            )
            logger.info("whisper_model_loaded")
        except Exception as e:
            logger.error("whisper_model_load_failed", error=str(e))
            raise
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        try:
            Path(settings.diarizer_cache_dir).mkdir(parents=True, exist_ok=True)
            logger.info("loading_diarizer", cache_dir=settings.diarizer_cache_dir)
            _diarizer = Pipeline.from_pretrained(
                settings.pyannote_protocol,
                cache_dir=settings.diarizer_cache_dir,
                use_auth_token=settings.huggingface_token
            )
            logger.info("diarizer_loaded")
        except ModelNotFoundError as e:
            logger.error("diarizer_model_not_found", error=str(e))
            raise
        except Exception as e:
            logger.error("diarizer_load_failed", error=str(e))
            raise
    return _diarizer

def vad_segment_audio(audio_path: Path):
    """Use webrtcvad to extract speech segments, fallback to fixed windows."""
    start_time = perf_counter()
    try:
        audio = AudioSegment.from_file(str(audio_path)).set_channels(1).set_frame_rate(16000)
        raw = audio.raw_data
        vad = webrtcvad.Vad(settings.vad_level)
        frames = [raw[i:i+320] for i in range(0, len(raw), 320)]
        segments, current, ts = [], None, 0.0
        for frame in frames:
            is_speech = vad.is_speech(frame, sample_rate=16000)
            if is_speech and current is None:
                current = ts
            if not is_speech and current is not None:
                segments.append((current, ts))
                current = None
            ts += 0.02
        if current is not None:
            segments.append((current, ts))
        if not segments:
            raise RuntimeError("no speech detected")
        return segments
    except Exception as e:
        logger.warning("vad_segmentation_failed", error=str(e))
        audio_length = AudioSegment.from_file(str(audio_path)).duration_seconds
        window = settings.segment_length_s
        return [(i, min(i + window, audio_length)) for i in range(0, int(audio_length), window)]
    finally:
        SEGMENTATION_DURATION.observe(perf_counter() - start_time)

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    TASK_RUN_COUNTER.labels(task_name="transcribe_segments").inc()
    whisper = get_whisper_model()
    src = Path(settings.upload_folder) / upload_id
    dst = Path(settings.results_folder) / upload_id
    dst.mkdir(parents=True, exist_ok=True)
    logger.info("transcription_start", upload_id=upload_id, path=str(src))

    segments = vad_segment_audio(src)
    results = []
    for idx, (start, end) in enumerate(segments):
        with TRANSCRIPTION_DURATION.time():
            try:
                res = whisper.transcribe(
                    str(src),
                    language="ru",
                    vad_filter=True,
                    word_timestamps=True,
                    offset=start,
                    duration=end - start
                )
                text = " ".join(seg["text"] for seg in res["segments"])
                results.append({"segment": idx, "start": start, "end": end, "text": text})
                logger.debug("segment_transcribed", idx=idx, start=start, end=end)
            except Exception as e:
                logger.error("segment_transcription_failed", idx=idx, error=str(e))
                results.append({"segment": idx, "start": start, "end": end, "text": "", "error": str(e)})

    out = dst / "transcript.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("transcription_complete", upload_id=upload_id, out=str(out))

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    TASK_RUN_COUNTER.labels(task_name="diarize_full").inc()
    diarizer = get_diarizer()
    src = Path(settings.upload_folder) / upload_id
    dst = Path(settings.results_folder) / upload_id
    dst.mkdir(parents=True, exist_ok=True)
    logger.info("diarization_start", upload_id=upload_id, path=str(src))

    try:
        diarization = diarizer(str(src))
        segments = [
            {"start": t.start, "end": t.end, "speaker": s}
            for t, _, s in diarization.itertracks(yield_label=True)
        ]
    except Exception as e:
        logger.error("diarization_failed", error=str(e))
        segments = []

    out = dst / "diarization.json"
    out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("diarization_complete", upload_id=upload_id, out=str(out))

@shared_task(name="tasks.cleanup_old_files")
def cleanup_old_files():
    """Cleanup old uploads and results beyond retention period."""
    TASK_RUN_COUNTER.labels(task_name="cleanup_old_files").inc()
    retention = settings.file_retention_days
    cutoff = datetime.utcnow() - timedelta(days=retention)

    # Очистка загруженных файлов
    for f in Path(settings.upload_folder).iterdir():
        if f.is_file() and datetime.utcfromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()

    # Очистка результатов
    for d in Path(settings.results_folder).iterdir():
        if d.is_dir() and datetime.utcfromtimestamp(d.stat().st_mtime) < cutoff:
            shutil.rmtree(d)

    logger.info("cleanup_old_files_complete", retention_days=retention)