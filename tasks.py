import os
import torch
import logging
import whisperx
import glob
import shutil
import time
from datetime import datetime, timedelta

from celery import shared_task, Task
from config.settings import settings


# Создание папки для загрузок на старте приложения
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)

# ==============================
# SUPPORTED FORMATS (расширяем по необходимости)
# ==============================
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".mp4", ".webm", ".opus", ".wma", ".alac", ".aiff"}

# ==============================
# SINGLETON HOLDERS
# ==============================
whisper_model = None
align_model = None
align_metadata = None
diarization_pipeline = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisperx.load_model(
            settings.WHISPER_MODEL,
            settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
    return whisper_model

def get_align_model(language_code):
    global align_model, align_metadata
    if align_model is not None and align_metadata is not None:
        return align_model, align_metadata
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=settings.DEVICE
        )
    except Exception as e:
        logging.warning(f"Align model for {language_code} could not be loaded: {e}")
        align_model, align_metadata = None, None
    return align_model, align_metadata

def get_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is not None:
        return diarization_pipeline
    try:
        from pyannote.audio import Pipeline
        diarization_pipeline = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    except Exception as e:
        logging.warning(f"Could not load diarization pipeline: {e}")
        diarization_pipeline = None
    return diarization_pipeline

def is_audio_file(filename):
    return os.path.splitext(filename)[1].lower() in AUDIO_EXTENSIONS

# ==============================
# MAIN TASK
# ==============================
@shared_task(bind=True, name='tasks.transcribe_task', max_retries=3, default_retry_delay=60)
def transcribe_task(self, file_path: str):
    try:
        # 1. Transcribe
        model = get_whisper_model()
        result = model.transcribe(file_path)
        segments = result.get("segments", [])
        lang = result.get("language", settings.LANGUAGE_CODE)

        # 2. Alignment (optional)
        align_model, align_metadata = get_align_model(lang)
        if align_model and align_metadata and segments:
            try:
                result = whisperx.align(
                    segments,
                    align_model,
                    align_metadata,
                    file_path,
                    settings.DEVICE
                )
            except Exception as e:
                logging.warning(f"Align failed: {e}")

        # 3. Diarization (optional)
        diarization = []
        diarization_pipeline = get_diarization_pipeline()
        if diarization_pipeline:
            try:
                diarization_raw = diarization_pipeline(file_path)
                diarization = [
                    {
                        "speaker": str(turn["label"]),
                        "start": float(turn["start"]),
                        "end": float(turn["end"])
                    }
                    for turn in diarization_raw.itertracks(yield_label=True)
                ]
            except Exception as e:
                logging.warning(f"Diarization failed: {e}")
                diarization = []

        # 4. Schedule file cleanup after 1 day (24 hours)
        cleanup_files.apply_async(kwargs={'file_path': file_path}, eta=datetime.utcnow() + timedelta(days=1))

        return {
            "segments": segments,
            "language": lang,
            "diarization": diarization,
            "file_path": file_path,
        }
    except Exception as exc:
        logging.error(f"Transcription failed: {exc}", exc_info=True)
        raise self.retry(exc=exc)

# ==============================
# CLEANUP TASK
# ==============================
@shared_task
def cleanup_files(file_path=None):
    """Удаляет конкретный файл или чистит все старые и ненужные файлы в папке uploads."""
    folder = settings.UPLOAD_FOLDER
    cutoff = time.time() - 24 * 3600  # 1 day ago

    # Если конкретный файл - просто удалить
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            meta = os.path.splitext(file_path)[0] + "_labels.json"
            if os.path.exists(meta):
                os.remove(meta)
        except Exception as e:
            logging.warning(f"Failed to remove file {file_path}: {e}")
        return

    # Иначе -- чистим все старое
    for root, dirs, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            if not is_audio_file(full_path) and not f.endswith("_labels.json"):
                continue
            try:
                if os.path.getmtime(full_path) < cutoff:
                    os.remove(full_path)
            except Exception as e:
                logging.warning(f"Failed to remove {full_path}: {e}")

    # Чистка если мало места
    if get_disk_free_ratio() < 0.25:
        logging.warning("Disk space critically low (<25%). Forcing additional cleanup.")
        remove_oldest_files(folder, keep_ratio=0.25)

def get_disk_free_ratio():
    """Возвращает долю свободного места (0..1) на диске с UPLOAD_FOLDER."""
    total, used, free = shutil.disk_usage(settings.UPLOAD_FOLDER)
    return free / total if total else 0.0

def remove_oldest_files(folder, keep_ratio=0.25):
    """Удаляет самые старые файлы, пока не освободится >= keep_ratio места."""
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            full_path = os.path.join(root, f)
            if is_audio_file(full_path) or f.endswith("_labels.json"):
                try:
                    mtime = os.path.getmtime(full_path)
                    files.append((mtime, full_path))
                except Exception:
                    continue
    files.sort()  # Старые первыми
    while get_disk_free_ratio() < keep_ratio and files:
        _, file_to_delete = files.pop(0)
        try:
            os.remove(file_to_delete)
        except Exception as e:
            logging.warning(f"Failed to remove {file_to_delete}: {e}")

# ==============================
# FILE FINDER BY TASK ID
# ==============================
def get_file_path_by_task_id(task_id: str):
    """
    Возвращает путь к файлу по task_id (ищет по маске).
    Например: uploads/2024-06-19/{task_id}_*.*
    """
    base_folder = settings.UPLOAD_FOLDER
    for date_folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, date_folder)
        if not os.path.isdir(folder_path):
            continue
        # Ищем по маске uuid_*.ext
        for f in os.listdir(folder_path):
            if f.startswith(task_id + "_"):
                return os.path.join(folder_path, f)
    # Если не нашли — глобальный поиск по маске
    for root, dirs, files in os.walk(base_folder):
        for f in files:
            if f.startswith(task_id + "_"):
                return os.path.join(root, f)
    return None