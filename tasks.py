import os
from celery_app import app
from config.settings import UPLOAD_FOLDER, WHISPER_MODEL, DEVICE, WHISPER_COMPUTE_TYPE
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# global singletons
_whisper_model = None
_diarizer = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            model_name=WHISPER_MODEL,
            device=DEVICE,
            device_index=0,
            compute_type=WHISPER_COMPUTE_TYPE,
            inter_threads=1,
            intra_threads=1,
            tensor_parallel=False,
            max_queued_batches=1,
            flash_attention=False,
        )
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(os.getenv("PYANNOTE_PROTOCOL"), use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
    return _diarizer

@app.task
def diarize_full(filepath: str):
    # run on CPU
    diarizer = get_diarizer()
    return diarizer(filepath)

@app.task
def transcribe_full(filepath: str):
    # first diarize
    diarization = diarize_full(filepath)
    # then hand off chunked transcription
    return tasks.transcribe_segments.delay(filepath)

@app.task
def transcribe_segments(filepath: str):
    model = get_whisper()
    # split file into segments based on diarization
    segments = model.transcribe(filepath, beam_size=5)
    return [ {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments ]