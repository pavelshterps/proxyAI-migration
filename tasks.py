import os
import logging
from typing import List, Tuple, Dict

from celery_app import celery_app
from config.settings import (
    HUGGINGFACE_TOKEN,
    PYANNOTE_PROTOCOL,
    WHISPER_MODEL_NAME,
    DEVICE,
    WHISPER_COMPUTE_TYPE
)
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import librosa

logger = logging.getLogger(__name__)

# Кешируем загрузку моделей
_diarizer = None
_model = None

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        logger.info("Loading diarization model '%s'", PYANNOTE_PROTOCOL)
        _diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN
        )
    return _diarizer

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        # Скачиваем модель в кэш один раз
        logger.info("Downloading Whisper model '%s' via HuggingFace Hub", WHISPER_MODEL_NAME)
        model_path = snapshot_download(
            WHISPER_MODEL_NAME,
            use_auth_token=HUGGINGFACE_TOKEN
        )
        logger.info(
            "Loading Whisper model from '%s' on %s (compute_type=%s)",
            model_path, DEVICE, WHISPER_COMPUTE_TYPE
        )
        _model = WhisperModel(
            model_path,
            device=DEVICE,
            device_index=0,
            compute_type=WHISPER_COMPUTE_TYPE,
            tensor_parallel=False
        )
    return _model

@celery_app.task(name="tasks.diarize_full", queue="preprocess_cpu")
def diarize_full(filepath: str) -> List[Tuple[float, float]]:
    """
    Делает диаризацию всего файла, возвращает сегменты,
    и ставит задачу транскрипции на GPU.
    """
    diarizer = get_diarizer()
    diarization = diarizer(filepath)
    segments = [(t.start, t.end) for t, _, _ in diarization.itertracks(yield_label=True)]
    # запустить транскрипцию в GPU-очереди
    transcribe_segments.apply_async((filepath, segments), queue="preprocess_gpu")
    return segments

@celery_app.task(name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(filepath: str, segments: List[Tuple[float, float]]) -> Dict:
    """
    Делит каждый сегмент на куски по 30 секунд и транскрибирует на GPU.
    Возвращает полный текст и удаляет файл.
    """
    model = get_model()
    texts: List[str] = []

    for start, end in segments:
        remaining = end - start
        offset = start
        while remaining > 0:
            chunk_dur = min(remaining, 30.0)
            audio, _ = librosa.load(
                filepath,
                sr=16000,
                offset=offset,
                duration=chunk_dur
            )
            segment_result, _ = model.transcribe(audio)
            texts.append(" ".join(s.text for s in segment_result))
            offset += chunk_dur
            remaining -= chunk_dur

    full_text = "\n".join(texts)
    result = {"text": full_text}

    # Удаляем файл после обработки, чтобы не разрастался диск
    try:
        os.remove(filepath)
    except Exception:
        logger.exception("Failed to delete %s", filepath)

    return result