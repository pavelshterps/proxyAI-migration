import os
import logging
from typing import List, Tuple, Dict

from celery_app import celery_app
from config.settings import (
    HUGGINGFACE_TOKEN,
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

# Ленивые глобальные инстансы
_diarizer = None
_model = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        logger.info("Loading speaker diarization pipeline...")
        _diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN
        )
    return _diarizer

def get_model():
    global _model
    if _model is None:
        logger.info(
            "Loading Whisper model '%s' on %s with compute_type=%s",
            WHISPER_MODEL_NAME, DEVICE, WHISPER_COMPUTE_TYPE
        )
        _model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
    return _model

@celery_app.task(name="tasks.diarize_full", queue="preprocess_cpu")
def diarize_full(filepath: str) -> List[Tuple[float, float]]:
    """
    Запускает полную диаризацию на CPU и ставит в очередь транскрипцию.
    Возвращает список сегментов (start, end).
    """
    logger.info("Diarization on CPU for %s", filepath)
    diarizer = get_diarizer()
    diarization = diarizer(filepath)
    segments = [
        (turn.start, turn.end)
        for turn, _, _ in diarization.itertracks(yield_label=True)
    ]
    # Далее — транскрипция на GPU
    transcribe_segments.apply_async((filepath, segments), queue="preprocess_gpu")
    return segments

@celery_app.task(name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(filepath: str, segments: List[Tuple[float, float]]) -> Dict:
    """
    Запускает Whisper chunked-транскрипцию на GPU для переданных сегментов.
    В конце удаляет файл.
    """
    logger.info("Transcription on GPU for %s (%d segments)", filepath, len(segments))
    result: Dict = {}
    try:
        model = get_model()
        texts = []
        for start, end in segments:
            audio, _ = librosa.load(filepath, sr=16000, offset=start, duration=end - start)
            segments_text, _ = model.transcribe(audio)
            texts.append(" ".join(seg.text for seg in segments_text))
        result = {"text": "\n".join(texts)}
    except Exception:
        logger.exception("Error during transcription for %s", filepath)
        result = {"error": "Transcription failed"}
    finally:
        # Убираем исходный файл
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
    return result