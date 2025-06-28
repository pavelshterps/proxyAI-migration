import os
import json
import logging
from typing import List, Dict

from celery import shared_task, Task
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

from config.settings import settings

logger = logging.getLogger(__name__)

# Ленивая инициализация моделей
_diarizer = None
_whisper_model = None


def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading diarizer into cache {cache_dir}...")
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_MODEL,
            cache_dir=cache_dir,
            use_auth_token=settings.HF_TOKEN or None,
            progress_bar=False,
        )
        logger.info("Diarizer loaded")
    return _diarizer


def get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL
        logger.info(f"Loading WhisperModel once at startup: {settings.dict(include={'WHISPER_MODEL','WHISPER_DEVICE','WHISPER_COMPUTE_TYPE','WHISPER_DEVICE_INDEX'})}")
        _whisper_model = WhisperModel(
            model_path,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=settings.WHISPER_DEVICE_INDEX,
        )
        logger.info("WhisperModel loaded (quantized format)")
    return _whisper_model


@shared_task(name="tasks.diarize_full")
def diarize_full(audio_path: str) -> List[Dict]:
    """
    Выполняет диаризацию всего файла и сразу отдаёт результат в качестве возвращаемого value task’а.
    """
    upload_id = os.path.splitext(os.path.basename(audio_path))[0]
    diarizer = get_diarizer()
    logger.info(f"Starting diarization for {audio_path}")
    diarization = diarizer(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
            "path": audio_path,  # важно для последующей транскрипции
        })
    logger.info(f"Diarization finished: {len(segments)} segments")
    return segments


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str, segments: List[Dict]) -> Dict:
    """
    Транскрибирует куски по результатам диаризации.
    """
    model = get_whisper()
    results = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        # faster-whisper: передаём сегмент целиком, а он нарежет
        result = model.transcribe(
            seg["path"],
            beam_size=settings.WHISPER_BEAM_SIZE,
            language=settings.WHISPER_MODEL_NAME,
            vad_parameters=None,
            word_timestamps=True,
            initial_prompt=None,
            best_of=settings.WHISPER_BEST_OF,
            start_ts=start,
            end_ts=end,
        )
        # собираем текст
        text = " ".join([w.text for w in result[0]])
        results.append({
            "start": start,
            "end": end,
            "speaker": seg["speaker"],
            "text": text,
        })
    # сохраняем весь json в результирующий backend
    output = {"segments": results}
    # celery backend сохраняет по ключу task_id, он равен upload_id
    logger.info(f"Transcription done for job {upload_id}")
    return output