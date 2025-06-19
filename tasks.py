import os
import logging

import whisperx
from celery_app import celery
from config.settings import settings

logger = logging.getLogger(__name__)

# Кэши моделей, чтобы не пере-загружать их на каждый вызов
_whisper_model = None
_align_model = None
_align_metadata = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        # Normalize DEVICE to lower-case for WhisperX
        device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
        # settings.WHISPER_MODEL берётся из .env, можно установить large-v3
        _whisper_model = whisperx.load_model(
            settings.WHISPER_MODEL,
            device,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
    return _whisper_model

def get_align_model():
    global _align_model, _align_metadata
    if _align_model is None or _align_metadata is None:
        device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code=settings.LANGUAGE_CODE,
            device=device
        )
    return _align_model, _align_metadata

@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def transcribe_task(self, audio_path: str):
    """
    1) Загружает модель (large-v3, если в settings.WHISPER_MODEL)
    2) Транскрибирует файл
    3) Выравнивает сегменты с помощью Pyannote
    4) Возвращает список сегментов для frontend
    """
    # Ещё раз нормализуем DEVICE
    device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE

    try:
        model = get_whisper_model()
        align_model, align_metadata = get_align_model()

        # 1) ASR
        result = model.transcribe(audio_path)

        # 2) Alignment (diarization)
        aligned = whisperx.align(
            result["segments"],
            model.tokenizer,
            audio_path,
            align_model,
            align_metadata,
            device=device
        )

        # 3) Формируем итоговый список сегментов
        segments = []
        for seg in aligned["segments"]:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "speaker": getattr(seg, "speaker", seg.speaker_label),
                "text": seg.text
            })

        return {
            "segments": segments,
            "audio_filepath": audio_path
        }

    except Exception as exc:
        logger.error(f"Transcription failed: {exc}")
        # Повторяем до 3 раз с задержкой в 60s
        raise self.retry(exc=exc)

@celery.task
def cleanup_files(path: str):
    """
    Удаляет временные файлы после обработки
    """
    try:
        os.remove(path)
    except OSError:
        pass