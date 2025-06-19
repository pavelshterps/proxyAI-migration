import os
import json
import shutil
import glob
import psutil
import whisperx
import torch
import datetime
from celery import shared_task, Task
from celery.utils.log import get_task_logger
from pyannote.audio import Pipeline
from config.settings import settings

logger = get_task_logger(__name__)

# Singleton-объекты (ленивая инициализация)
_whisper_model = None
_align_model = None
_align_metadata = None
_diarization_pipeline = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
        _whisper_model = whisperx.load_model(
            settings.WHISPER_MODEL,
            device,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
    return _whisper_model

def get_align_model():
    global _align_model, _align_metadata
    if _align_model is None:
        try:
            device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
            _align_model, _align_metadata = whisperx.load_align_model(
                language_code=settings.LANGUAGE_CODE,
                device=device
            )
        except ValueError:
            logger.warning(f"No align model for {settings.LANGUAGE_CODE}, skipping")
            _align_model, _align_metadata = None, None
    return _align_model, _align_metadata

def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        _diarization_pipeline = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _diarization_pipeline

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def transcribe_task(self, file_path: str):
    """
    Транскрипция + выравнивание + диаризация.
    Результат сохраняется в JSON-файлы рядом с аудио.
    """
    try:
        # 1) ASR
        model = get_whisper_model()
        result = model.transcribe(file_path)
        segments = result["segments"]
        lang = result.get("language")

        # 2) Forced-alignment (если есть модель)
        align_model, align_metadata = get_align_model()
        if align_model:
            result = whisperx.align(
                segments,
                align_model,
                align_metadata,
                file_path,
                settings.DEVICE
            )
            segments = result["segments"]

        # 3) Speaker diarization
        diarizer = get_diarization_pipeline()
        diarization = diarizer(file_path)

        # 4) Собираем вывод
        output = {
            "file_path": file_path,
            "language": lang,
            "transcription": [seg._asdict() for seg in segments],
            "diarization": [
                {"start": turn.start, "end": turn.end, "speaker": turn.label}
                for turn in diarization.get_timeline()
            ]
        }

        # 5) Сохраняем в файлы
        base = os.path.splitext(file_path)[0]
        with open(f"{base}_transcript.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return output

    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        raise self.retry(exc=exc)

@shared_task
def cleanup_files():
    """
    Удаляет:
     - все файлы старше FILE_RETENTION_DAYS
     - запускается автоматически при низком свободном диске
    """
    folder = settings.UPLOAD_FOLDER
    retention = settings.FILE_RETENTION_DAYS
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=retention)

    # 1) Удаляем старые
    for path in glob.glob(os.path.join(folder, "*", "*_*")):
        try:
            mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                os.remove(path)
                logger.info(f"Removed old file: {path}")
        except Exception:
            logger.exception(f"Error removing {path}")

    # 2) Если свободного места < 25%, шорткат: досрочный запуск
    usage = psutil.disk_usage(folder)
    free_pct = usage.free / usage.total * 100
    if free_pct < 25:
        logger.warning(f"Low disk space ({free_pct:.1f}%), re-running cleanup")
        cleanup_files.delay()

def get_file_path_by_task_id(task_id: str) -> str | None:
    """
    Находит файл по task_id, основываясь на том, что мы сохраняем его путь в результатах.
    """
    # здесь предполагаем, что Celery хранит результат {"file_path": ...}
    from celery.result import AsyncResult
    res = AsyncResult(task_id)
    data = res.result or {}
    return data.get("file_path")