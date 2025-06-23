import os
import glob
import logging
from typing import Dict

from celery_app import celery_app
from config.settings import (
    HUGGINGFACE_TOKEN,
    HF_CACHE_DIR,
    PYANNOTE_PROTOCOL,
    WHISPER_MODEL_NAME,
    WHISPER_COMPUTE_TYPE,
    DEVICE,
)
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import librosa

logger = logging.getLogger(__name__)

# 1) Глобальная инициализация на старте воркера:
try:
    diarizer = Pipeline.from_pretrained(
        PYANNOTE_PROTOCOL,
        use_auth_token=HUGGINGFACE_TOKEN,
        cache_dir=HF_CACHE_DIR
    )
    logger.info("Loaded diarization pipeline")
except Exception as e:
    logger.exception("Failed to load diarization pipeline: %s", e)
    diarizer = None

try:
    model = WhisperModel(
        WHISPER_MODEL_NAME,
        device=DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE
    )
    logger.info("Loaded WhisperModel")
except Exception as e:
    logger.exception("Failed to load WhisperModel: %s", e)
    model = None


@celery_app.task(name="tasks.transcribe_full", queue="preprocess")
def transcribe_full(filepath: str) -> Dict:
    """
    1) speaker diarization
    2) load audio segments via librosa
    3) whisper transcription
    Clean up files after.
    """
    logger.info("Starting full transcription: %s", filepath)

    # 2) Проверка, что модели инициализированы
    if diarizer is None or model is None:
        err = "Models not initialized"
        logger.error(err)
        return {"error": err}

    try:
        # Сперва диаризация
        diarization = diarizer(filepath)
        segments = [
            (turn.start, turn.end)
            for turn, _, _ in diarization.itertracks(yield_label=True)
        ]

        # Затем транскрипция каждого сегмента
        full_text = []
        for start, end in segments:
            audio, sr = librosa.load(
                filepath, sr=16000, offset=start, duration=end - start
            )
            transcribed_segments, _ = model.transcribe(audio)
            full_text.append(" ".join(seg.text for seg in transcribed_segments))

        result = {"text": "\n".join(full_text)}

    except Exception as e:
        logger.exception("Error in transcription pipeline")
        result = {"error": str(e)}

    # Cleanup
    try:
        for f in glob.glob("/tmp/chunks/*.wav"):
            os.remove(f)
        os.remove(filepath)
    except Exception:
        pass

    return result