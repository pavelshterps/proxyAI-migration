import os
import glob
import logging
from typing import Dict

from celery_app import celery_app
from config.settings import (
    HUGGINGFACE_TOKEN,
    HF_CACHE_DIR,
    UPLOAD_FOLDER,
    PYANNOTE_PROTOCOL,
    WHISPER_MODEL_NAME,
    DEVICE,
    WHISPER_COMPUTE_TYPE,
)
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import librosa

logger = logging.getLogger(__name__)

# Глобальные переменные для ленивой инициализации
_diarizer = None
_model = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        logger.info("Loading speaker diarization pipeline...")
        _diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN,
            cache_dir=HF_CACHE_DIR
        )
    return _diarizer

def get_model():
    global _model
    if _model is None:
        logger.info(
            "Loading Whisper model '%s' on device %s with compute type %s...",
            WHISPER_MODEL_NAME, DEVICE, WHISPER_COMPUTE_TYPE
        )
        _model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
    return _model

@celery_app.task(name="tasks.transcribe_full")
def transcribe_full(filepath: str) -> Dict:
    logger.info("Starting full transcription for file: %s", filepath)
    try:
        diarizer = get_diarizer()
        model = get_model()

        # 1) Speaker diarization
        diarization = diarizer(filepath)
        segments = [
            (turn.start, turn.end)
            for turn, _, _ in diarization.itertracks(yield_label=True)
        ]

        # 2) Transcription per segment
        transcript_parts = []
        for start, end in segments:
            audio, sr = librosa.load(
                filepath,
                sr=16000,
                offset=start,
                duration=end - start
            )
            segments_text, _ = model.transcribe(audio)
            text = " ".join(seg.text for seg in segments_text)
            transcript_parts.append(text)

        return {"text": "\n".join(transcript_parts)}

    except Exception as e:
        logger.exception("Error in transcription pipeline for %s", filepath)
        return {"error": str(e)}

    finally:
        # Clean up uploaded file and any temp chunks
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            for chunk in glob.glob(os.path.join(UPLOAD_FOLDER, "*.wav")):
                os.remove(chunk)
        except Exception:
            pass