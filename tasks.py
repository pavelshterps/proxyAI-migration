import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

# === singletons ===
_whisper: WhisperModel | None = None
_diarizer: Pipeline | None = None

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        cfg = settings
        logger.info(f"Loading WhisperModel: path={cfg.WHISPER_MODEL_PATH} "
                    f"device={cfg.WHISPER_DEVICE} compute={cfg.WHISPER_COMPUTE_TYPE}")
        _whisper = WhisperModel(
            cfg.WHISPER_MODEL_PATH,
            device=cfg.WHISPER_DEVICE,
            compute_type=cfg.WHISPER_COMPUTE_TYPE,
            device_index=cfg.WHISPER_DEVICE_INDEX,
            # можно задать pool-threads через intra/inter, если нужно
        )
        logger.info("WhisperModel loaded")
    return _whisper

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading diarizer into {cache_dir}")
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_MODEL,
            cache_dir=cache_dir
        )
        logger.info("Diarizer loaded")
    return _diarizer

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    cfg = settings
    model = get_whisper()

    src = Path(cfg.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(cfg.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Transcribe] {src}")
    # TODO: если понадобится сегментация по беззвучию — здесь её вставить
    segments = [(0.0, None)]

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"Segment {idx}: {start}-{end}")
        res = model.transcribe(
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
            "segment": idx, "start": start, "end": end, "text": text
        })

    out = dst_dir / "transcript.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    logger.info(f"Transcript saved: {out}")

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    cfg = settings
    diarizer = get_diarizer()

    src = Path(cfg.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(cfg.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Diarize] {src}")
    diarization = diarizer(str(src))

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start, "end": turn.end, "speaker": speaker
        })

    out = dst_dir / "diarization.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)

    logger.info(f"Diarization saved: {out}")