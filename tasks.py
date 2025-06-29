import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

logger = logging.getLogger(__name__)

# Singleton instances, будут загружены только один раз на worker
_whisper_model: WhisperModel = None
_diarizer: Pipeline = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        model_path = os.getenv(
            "WHISPER_MODEL_PATH",
            "/hf_cache/models--guillaumekln--faster-whisper-medium"
        )
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        logger.info(
            f"⏳ Loading WhisperModel: "
            f"{{path: {model_path}, device: {device}, compute: {compute_type}}}"
        )
        _whisper_model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
        logger.info("✅ WhisperModel loaded")
    return _whisper_model


def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"⏳ Loading pyannote diarizer into cache: {cache_dir}")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=cache_dir
        )
        logger.info("✅ Diarizer loaded")
    return _diarizer


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    """
    Транскрипция аудио целиком с разбивкой на сегменты по тикам Whisper.
    Сохраняет RESULTS_FOLDER/{upload_id}/transcript.json
    """
    whisper = get_whisper_model()
    audio_path = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"▶️  Transcribing upload={upload_id}: {audio_path}")
    try:
        result = whisper.transcribe(
            str(audio_path),
            beam_size=int(os.getenv("WHISPER_BEAM_SIZE", 5)),
            language=os.getenv("WHISPER_LANGUAGE", "ru"),
            word_timestamps=True
        )
    except Exception as e:
        logger.exception(f"❌ Whisper transcribe failed for {upload_id}")
        raise

    transcript = []
    for seg in result["segments"]:
        transcript.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    out_file = out_dir / "transcript.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Transcription saved to {out_file}")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    """
    Диаризация полного аудио.
    Сохраняет RESULTS_FOLDER/{upload_id}/diarization.json
    """
    diarizer = get_diarizer()
    audio_path = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"▶️  Diarizing upload={upload_id}: {audio_path}")
    try:
        diarization = diarizer(str(audio_path))
    except Exception as e:
        logger.exception(f"❌ Diarization failed for {upload_id}")
        raise

    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    out_file = out_dir / "diarization.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(speakers, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Diarization saved to {out_file}")