# tasks.py
import os
import json
import shutil
import logging
from datetime import datetime, timedelta

from celery import Celery, signals
from celery.schedules import crontab

from crud import (
    get_user_by_api_key,
    create_upload_record,
    get_upload_for_user,
    update_upload_status
)
from config.settings import settings

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Инициализация логгера
logger = logging.getLogger(__name__)

# Настройка Celery
celery_app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Планировщик периодических задач
celery_app.conf.beat_schedule = {
    "cleanup-old-uploads": {
        "task": "tasks.cleanup_old_uploads",
        "schedule": crontab(hour="*/1"),  # каждый час
    },
}

# Singleton модели Whisper и Pyannote
_whisper: WhisperModel | None = None
_pyannote: Pipeline | None = None

def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper:
        return _whisper

    model_path = settings.WHISPER_MODEL_PATH  # локальный путь к quantized-модели
    init_kwargs = {
        "model_size_or_path": model_path,
        "device": "cuda",
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
        "batch_size": settings.WHISPER_BATCH_SIZE,
        "cache_dir": settings.HUGGINGFACE_CACHE_DIR,
        "local_files_only": True,  # не скачиваем, если есть локально
    }

    logger.info(
        "loading whisper model",
        extra={"model_path": model_path, **init_kwargs}
    )
    _whisper = WhisperModel(**init_kwargs)
    return _whisper

def get_pyannote() -> Pipeline:
    global _pyannote
    if _pyannote:
        return _pyannote

    pipeline_name = settings.PYANNOTE_PIPELINE
    _pyannote = Pipeline.from_pretrained(
        pipeline_name,
        use_auth_token=None,
        cache_dir=settings.HUGGINGFACE_CACHE_DIR
    )
    return _pyannote

@celery_app.task(name="tasks.preprocess_and_diarize")
def preprocess_and_diarize(
    file_path: str,
    upload_id: str,
    speaker_map: dict | None = None
) -> str:
    """
    1. Диаризация (CPU).
    2. Возвращает путь к JSON-файлу с сегментами.
    """
    try:
        # Обновляем статус в БД
        # (предполагается, что сессия передана через бэкэнд или global Session)
        session = celery_app.backend.session_cls()
        update_upload_status(session, upload_id, "diarizing")

        pipeline = get_pyannote()
        diarization = pipeline({"uri": upload_id, "audio": file_path})

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        # Замена имён по speaker_map
        if speaker_map:
            for seg in segments:
                seg["speaker"] = speaker_map.get(seg["speaker"], seg["speaker"])

        out_path = f"{file_path}.diarization.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)

        update_upload_status(session, upload_id, "diarized")
        return out_path

    except Exception as e:
        logger.exception("Ошибка во время diarization")
        update_upload_status(session, upload_id, "failed")
        raise

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(
    file_path: str,
    diarization_json: str,
    upload_id: str,
    speaker_map: dict | None = None
) -> str:
    """
    1. Транскрибируем каждый сегмент (GPU).
    2. Собираем единый JSON результата.
    """
    try:
        session = celery_app.backend.session_cls()
        update_upload_status(session, upload_id, "transcribing")

        whisper = get_whisper()
        with open(diarization_json, "r", encoding="utf-8") as f:
            segments = json.load(f)

        results = []
        for seg in segments:
            transcription, _ = whisper.transcribe(
                file_path,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
                segment=seg,
            )
            text = " ".join([w["word"] for w in transcription])
            speaker = seg["speaker"]
            if speaker_map:
                speaker = speaker_map.get(speaker, speaker)

            results.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker,
                "text": text
            })

        out_path = f"{file_path}.result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        update_upload_status(session, upload_id, "completed")
        return out_path

    except Exception:
        logger.exception("Ошибка во время транскрипции")
        update_upload_status(session, upload_id, "failed")
        raise

@celery_app.task(name="tasks.cleanup_old_uploads")
def cleanup_old_uploads() -> None:
    """
    Удаляет записи в БД и файлы старше retention_period.
    Запускается по расписанию каждый час.
    """
    try:
        session = celery_app.backend.session_cls()
        cutoff = datetime.utcnow() - timedelta(hours=settings.UPLOAD_RETENTION_HOURS)
        # Предполагаем, что Upload.expires_at хранит время истечения
        uploads = session.query(Upload).filter(Upload.expires_at < cutoff).all()
        for upload in uploads:
            # удаляем файлы на диске
            base = os.path.splitext(upload.filename)[0]
            for ext in (".diarization.json", ".result.json", ""):
                path = f"{base}{ext}"
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    except Exception:
                        logger.warning(f"Не удалось удалить файл {path}", exc_info=True)
            # удаляем запись из БД
            session.delete(upload)
        session.commit()
        logger.info("cleanup_old_uploads выполнен")
    except Exception:
        logger.exception("Ошибка в cleanup_old_uploads")
        raise

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    При запуске воркера: прогрев моделей
    """
    try:
        get_pyannote()
        get_whisper()
    except Exception:
        logger.exception("Ошибка при preload_and_warmup")
        raise