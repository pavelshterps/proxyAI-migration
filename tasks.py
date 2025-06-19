import os
import logging

import whisperx
from celery_app import celery
from celery.result import AsyncResult
from config.settings import settings

logger = logging.getLogger(__name__)

# Cache models to avoid reloading on every task
_whisper_model = None
_align_model = None
_align_metadata = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        # Normalize DEVICE to lower-case for WhisperX
        device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
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
    1) Load WhisperX model (e.g., large-v3 via settings.WHISPER_MODEL)
    2) Transcribe the audio file
    3) Align (diarize) segments with Pyannote
    4) Return segments list for frontend
    """
    device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
    try:
        model = get_whisper_model()
        align_model, align_metadata = get_align_model()

        # 1) Automatic speech recognition
        result = model.transcribe(audio_path)

        # 2) Alignment/diarization
        aligned = whisperx.align(
            result['segments'],
            model.tokenizer,
            audio_path,
            align_model,
            align_metadata,
            device=device
        )

        # 3) Format segments
        segments = []
        for seg in aligned['segments']:
            segments.append({
                'start': seg.start,
                'end': seg.end,
                'speaker': getattr(seg, 'speaker', seg.speaker_label),
                'text': seg.text,
            })

        return {
            'segments': segments,
            'audio_filepath': audio_path
        }
    except Exception as exc:
        logger.error(f"Transcription failed: {exc}")
        # Retry up to 3 times with 60s delay
        raise self.retry(exc=exc)

@celery.task
def cleanup_files(path: str):
    """
    Remove temporary files after processing
    """
    try:
        os.remove(path)
    except OSError:
        pass


def get_file_path_by_task_id(task_id: str) -> str:
    """
    Retrieve the audio file path for a completed transcription task.
    Returns the file path if the task succeeded, or None if not.
    """
    res = AsyncResult(task_id, app=celery)
    if res.state == 'SUCCESS':
        payload = res.get()
        return payload.get('audio_filepath')
    return None