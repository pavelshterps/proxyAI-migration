import os
import glob
import shutil
import json
import time
import logging
import torch
import whisperx
from pyannote.audio import Pipeline

from celery_app import celery
from config.settings import settings

# ----- Configuration -----
UPLOAD_DIR = settings.UPLOAD_FOLDER
RETENTION_DAYS = settings.FILE_RETENTION_DAYS
DEVICE = settings.DEVICE.lower()  # must be "cpu" or "cuda"
# --------------------------

# ---- Lazy singletons ----
_whisper_model = None
_align_model = None
_align_metadata = None
_diarization_pipeline = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisperx.load_model(
            settings.WHISPER_MODEL,
            DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
    return _whisper_model

def get_align_model():
    global _align_model, _align_metadata
    if _align_model is None:
        try:
            _align_model, _align_metadata = whisperx.load_align_model(
                language_code=settings.LANGUAGE_CODE,
                device=DEVICE
            )
        except ValueError as e:
            logging.warning(f"No align model for language {settings.LANGUAGE_CODE}: {e}")
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

def get_file_path_by_task_id(task_id: str) -> str:
    """
    Finds the uploaded file path for a given Celery task ID.
    Assumes files are named <task_id>_<originalname> inside UPLOAD_DIR/YYYY-MM-DD/.
    """
    pattern = os.path.join(UPLOAD_DIR, "*", f"{task_id}_*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

# ---- Transcription Task ----
@celery.task(bind=True, name="tasks.transcribe_task", max_retries=3, default_retry_delay=60)
def transcribe_task(self, file_path: str):
    try:
        # 1) ASR
        model = get_whisper_model()
        result = model.transcribe(file_path)
        segments = result["segments"]
        language = result.get("language")

        # 2) Forced alignment (if available)
        align_model, align_metadata = get_align_model()
        if align_model is not None:
            aligned = whisperx.align(
                segments,
                align_model,
                align_metadata,
                file_path,
                device=DEVICE
            )
            segments = aligned["segments"]

        # 3) Diarization
        diarizer = get_diarization_pipeline()
        diar = diarizer(file_path)
        diarization = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            diarization.append({
                "start": turn.start,
                "end":   turn.end,
                "speaker": speaker
            })

        # 4) Save a default labels.json (identity mapping)
        labels_path = os.path.splitext(file_path)[0] + "_labels.json"
        mapping = {d["speaker"]: d["speaker"] for d in diarization}
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

        return {
            "file_path":   file_path,
            "language":    language,
            "segments":    segments,
            "diarization": diarization
        }

    except Exception as exc:
        logging.exception("Transcription failed")
        raise self.retry(exc=exc)

# ---- Cleanup Task ----
@celery.task(name="tasks.cleanup_files")
def cleanup_files():
    """
    Removes files older than RETENTION_DAYS, prunes empty folders,
    and if disk free <25%, reschedules itself immediately.
    """
    now = time.time()
    cutoff = now - RETENTION_DAYS * 86400

    # Walk UPLOAD_DIR
    for root, dirs, files in os.walk(UPLOAD_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except Exception:
                logging.warning(f"Could not remove {path}", exc_info=True)
        # if folder now empty, remove it
        if not os.listdir(root):
            try:
                shutil.rmtree(root)
            except Exception:
                logging.debug(f"Could not remove dir {root}", exc_info=True)

    # If disk nearly full, re-enqueue cleanup
    stat = os.statvfs(UPLOAD_DIR)
    free_ratio = stat.f_bavail / stat.f_blocks
    if free_ratio < 0.25:
        cleanup_files.delay()