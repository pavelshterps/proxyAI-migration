# tasks.py
import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

# Singleton-объекты
_whisper_model: WhisperModel | None = None
_diarizer: Pipeline | None = None

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            f"Loading WhisperModel once at startup: "
            f"{{
                'model_path': settings.WHISPER_MODEL_PATH,
                'device': settings.WHISPER_DEVICE,
                'compute_type': settings.WHISPER_COMPUTE_TYPE
            }}"
        )
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=settings.WHISPER_DEVICE_INDEX,
        )
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading diarizer into cache {cache_dir}...")
        os.environ["HUGGINGFACE_TOKEN"] = settings.HUGGINGFACE_TOKEN
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            cache_dir=cache_dir,
        )
        logger.info("Diarizer loaded")
    return _diarizer

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for {src}")
    # один сегмент — вся дорожка
    result = whisper.transcribe(
        str(src),
        beam_size=settings.WHISPER_BEAM_SIZE,
        language="ru",
        word_timestamps=True,
    )

    # собираем текст
    transcript = [
        {
            "segment": idx,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for idx, seg in enumerate(result["segments"])
    ]

    out_path = dst_dir / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Transcription saved to {out_path}")

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for {src}")
    diarization = diarizer(str(src))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Diarization saved to {out_path}")