import os, time, logging, shutil, re, json
import torch, whisperx
from pyannote.audio import Pipeline
from celery_app import app
from datetime import datetime
from config.settings import settings

# Use lazy-loading for whisper model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
whisper_model = None

def get_whisper_model():
    """
    Lazy-load the whisper model on first request to avoid OOM at startup.
    """
    global whisper_model
    if whisper_model is None:
        whisper_model = whisperx.load_model(
            settings.WHISPER_MODEL,
            DEVICE,
            compute_type="int8"
        )
    return whisper_model

align_model, metadata = whisperx.load_align_model(language_code=None, device=DEVICE)
diarization_pipeline = Pipeline.from_pretrained(settings.PYANNOTE_PROTOCOL, use_auth_token=settings.HUGGINGFACE_TOKEN)

@app.task(bind=True, name='tasks.transcribe_task', max_retries=3, default_retry_delay=60)
def transcribe_task(self, file_path: str):
    try:
        model = get_whisper_model()
        result = model.transcribe(file_path)
        # Alignment
        segments = result['segments']
        aligned = whisperx.align(segments, align_model, metadata, file_path, DEVICE)
        # Diarization
        diar = diarization_pipeline(file_path)
        diar_out = [
            {'start': seg.start, 'end': seg.end, 'speaker': spk}
            for seg, _, spk in diar.itertracks(yield_label=True)
        ]
        return {
            'text': result['text'],
            'segments': [s._asdict() for s in aligned],
            'diarization': diar_out,
            'file_path': file_path
        }
    except Exception as exc:
        logging.error(f"Transcribe error: {exc}", exc_info=True)
        cleanup_files.delay()
        raise self.retry(exc=exc)

@app.task(name='tasks.cleanup_files')
def cleanup_files():
    base = settings.UPLOAD_FOLDER
    retention = settings.FILE_RETENTION_DAYS
    total, used, free = shutil.disk_usage(base)
    low_disk = free / total < 0.05
    dates = sorted(d for d in os.listdir(base) if re.match(r'\d{4}-\d{2}-\d{2}', d))
    removed = 0
    for d in dates:
        dpath = os.path.join(base, d)
        folder_date = datetime.strptime(d, '%Y-%m-%d')
        age = (datetime.utcnow() - folder_date).days
        if age > retention or (low_disk and removed < 1):
            shutil.rmtree(dpath, ignore_errors=True)
            removed += 1
        if low_disk and removed >= 1:
            break
    return {'removed': removed}

def get_file_path_by_task_id(task_id: str):
    from celery.result import AsyncResult
    res = AsyncResult(task_id)
    if res.status == 'SUCCESS':
        return res.result.get('file_path')
    return None
