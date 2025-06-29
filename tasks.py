# tasks.py
import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import (
    UPLOAD_FOLDER, RESULTS_FOLDER,
    DIARIZER_CACHE_DIR,
    WHISPER_MODEL_PATH, WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE, WHISPER_DEVICE_INDEX,
    PYANNOTE_MODEL
)

logger = logging.getLogger(__name__)

_whisper: WhisperModel | None = None
_diarizer: Pipeline | None = None

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        logger.info(
            f"Loading WhisperModel: "
            f"path={WHISPER_MODEL_PATH} device={WHISPER_DEVICE} "
            f"compute={WHISPER_COMPUTE_TYPE}"
        )
        _whisper = WhisperModel(
            WHISPER_MODEL_PATH,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            device_index=WHISPER_DEVICE_INDEX
        )
        logger.info("WhisperModel loaded")
    return _whisper

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        os.makedirs(DIARIZER_CACHE_DIR, exist_ok=True)
        logger.info(f"Loading diarizer into cache {DIARIZER_CACHE_DIR}")
        _diarizer = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            cache_dir=DIARIZER_CACHE_DIR
        )
        logger.info("Diarizer loaded")
    return _diarizer

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    whisper = get_whisper()
    logger.info(f"Transcribing {src}")
    # пока весь файл одним куском
    segments = [(0.0, None)]
    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"Segment {idx}: {start}-{end}")
        res = whisper.transcribe(
            str(src),
            beam_size=5,
            language="ru",
            vad_filter=True,
            word_timestamps=True,
            offset=start,
            duration=None if end is None else (end - start),
        )
        text = res["segments"][0]["text"]
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text
        })

    out = dst_dir / "transcript.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    logger.info(f"Transcript saved to {out}")

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    diarizer = get_diarizer()
    logger.info(f"Diarizing {src}")
    di = diarizer(str(src))

    turns = []
    for turn, _, speaker in di.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    out = dst_dir / "diarization.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)
    logger.info(f"Diarization saved to {out}")