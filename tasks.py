import os
import logging
from pathlib import Path
import json

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

logger = logging.getLogger(__name__)

# Singletonы моделей
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = os.getenv(
            "WHISPER_MODEL_PATH",
            "/hf_cache/models--guillaumekln--faster-whisper-medium"
        )
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        logger.info(
            f"Loading WhisperModel once at startup: "
            f"{{'model': '{model_path}', 'device': '{device}', 'compute_type': '{compute_type}'}}"
        )
        _whisper_model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
        logger.info("WhisperModel loaded (quantized int8)")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache {cache_dir}…")
        # Убрали deprecated progress_bar, передаём только cache_dir
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=cache_dir
        )
        logger.info("Diarizer loaded")
    return _diarizer


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    audio_file = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Transcribe] {upload_id} → {audio_file}")
    # единый сегмент на всё — можно разбить по VAD
    segments = [(0.0, None)]

    transcript = []
    for i, (start, end) in enumerate(segments):
        logger.debug(f" segment {i}: {start}-{end}")
        result = whisper.transcribe(
            str(audio_file),
            beam_size=5,
            language="ru",
            vad_filter=True,
            word_timestamps=True,
            offset=start,
            duration=None if end is None else (end - start),
        )
        text = result["segments"][0]["text"]
        transcript.append({"segment": i, "start": start, "end": end, "text": text})

    with open(out_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    logger.info(f"[Transcribe] DONE → {out_dir / 'transcript.json'}")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    audio_file = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Diarize ] {upload_id} → {audio_file}")
    diarization = diarizer(str(audio_file))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    with open(out_dir / "diarization.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    logger.info(f"[Diarize ] DONE → {out_dir / 'diarization.json'}")