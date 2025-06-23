# tasks.py

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

# WhisperModel global instantiation without cache_dir
model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)

@celery_app.task(name="tasks.transcribe_full", queue="preprocess")
def transcribe_full(filepath: str) -> Dict:
    """
    Full pipeline in one task:
      1) speaker diarization
      2) load audio segments via librosa
      3) whisper transcription
    Clean up files after.
    """
    logger.info("Starting full transcription: %s", filepath)
    try:
        # Initialize diarizer inside task to delay HF download
        diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN,
            cache_dir=HF_CACHE_DIR
        )

        # 1. Diarize
        diarization = diarizer(filepath)
        segments = [
            (turn.start, turn.end)
            for turn, _, _ in diarization.itertracks(yield_label=True)
        ]

        # 2. Transcribe each segment
        full_text = []
        for start, end in segments:
            audio, sr = librosa.load(
                filepath, sr=16000, offset=start, duration=end - start
            )
            transcribed_segments, _ = model.transcribe(audio)
            full_text.append(
                " ".join(seg.text for seg in transcribed_segments)
            )

        result = {"text": "\n".join(full_text)}
    except Exception as e:
        logger.exception("Error in transcription pipeline")
        result = {"error": str(e)}

    # Cleanup: remove any leftover chunk files and the original file
    try:
        for f in glob.glob("/tmp/chunks/*.wav"):
            os.remove(f)
        os.remove(filepath)
    except Exception:
        pass

    return result
# tasks.py

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

# Instantiate WhisperModel without cache_dir to avoid incompatible constructor args
model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)

@celery_app.task(name="tasks.transcribe_full", queue="preprocess")
def transcribe_full(filepath: str) -> Dict:
    """
    Full pipeline in one task:
      1) speaker diarization
      2) load audio segments via librosa
      3) whisper transcription
    Clean up files after.
    """
    logger.info("Starting full transcription: %s", filepath)
    try:
        # 1. Initialize diarizer inside task
        diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN,
            cache_dir=HF_CACHE_DIR
        )
        diarization = diarizer(filepath)
        segments = [
            (turn.start, turn.end)
            for turn, _, _ in diarization.itertracks(yield_label=True)
        ]

        # 2. Transcribe each segment
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

    # Cleanup temporary files
    try:
        for f in glob.glob("/tmp/chunks/*.wav"):
            os.remove(f)
        os.remove(filepath)
    except Exception:
        pass

    return result