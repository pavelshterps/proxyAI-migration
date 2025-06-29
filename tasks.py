import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

# singletons
_whisper = None
_diarizer = None


def get_whisper():
    global _whisper
    if _whisper is None:
        logger.info(f"Loading WhisperModel: path={settings.WHISPER_MODEL_PATH} "
                    f"device={settings.WHISPER_DEVICE}[{settings.WHISPER_DEVICE_INDEX}], "
                    f"compute={settings.WHISPER_COMPUTE_TYPE}")
        _whisper = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            device_index=settings.WHISPER_DEVICE_INDEX,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            num_threads=settings.WHISPER_INTER_THREADS
        )
        logger.info("WhisperModel loaded")
    return _whisper


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache {cache}")
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            cache_dir=cache
        )
        logger.info("Diarizer loaded")
    return _diarizer


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {src}")
    # for now, one full-chunk segment
    result = whisper.transcribe(
        str(src),
        beam_size=settings.WHISPER_BEAM_SIZE,
        language="ru",
        vad_filter=True,
        word_timestamps=True,
    )
    segments = [{"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in result["segments"]]

    out = dst_dir / "transcript.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved transcript to {out}")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Diarizing {src}")
    diar = diarizer(str(src))
    segments = [{"start": turn.start, "end": turn.end, "speaker": spk}
                for turn, _, spk in diar.itertracks(yield_label=True)]

    out = dst_dir / "diarization.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved diarization to {out}")