import os
from celery_app import celery_app
from config.settings import settings

# ленивая загрузка
_diarizer = None
def get_diarizer():
    global _diarizer
    if _diarizer is None:
        from pyannote.audio import Pipeline
        _diarizer = Pipeline.from_pretrained(settings.PYANNOTE_PROTOCOL,
                                             use_auth_token=settings.HUGGINGFACE_TOKEN)
    return _diarizer

_model = None
def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=0,
            inter_threads=1,
            intra_threads=1,
        )
    return _model

@celery_app.task(name="tasks.diarize_full")
def diarize_full(path: str):
    diarizer = get_diarizer()
    diarization = diarizer(path)
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for speaker, turn in diarization.itertracks(yield_label=True)
    ]
    return segments

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(path: str):
    model = get_model()
    segments = []
    for i, chunk in enumerate(model.transcribe(path, beam_size=settings.ALIGN_BEAM_SIZE, word_timestamps=True)):
        segments.append({
            "id": i,
            "start": chunk.start,
            "end": chunk.end,
            "text": chunk.text.strip(),
        })
    return segments