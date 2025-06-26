from celery_app import celery_app
from config.settings import settings

_whisper_model = None
_pyannote_pipeline = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            inter_threads=1,
            intra_threads=1,
            chunk_length_s=settings.CHUNK_LENGTH_S
        )
    return _whisper_model

def get_pyannote_pipeline():
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        from pyannote.audio import Pipeline
        _pyannote_pipeline = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _pyannote_pipeline

@celery_app.task(name="tasks.diarize_full")
def diarize_full(path: str):
    pipeline = get_pyannote_pipeline()
    diarization = pipeline(path)
    segments = [
        (seg.start, seg.end)
        for seg in diarization.get_timeline().support()
    ]
    # dispatch transcription of each 30s segment
    result_tasks = [
        transcribe_segments.s(path, start, end)
        for start, end in segments
    ]
    # group and run in GPU queue
    group = celery_app.Group(*result_tasks).set(queue="preprocess_gpu")
    return group().get()

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(path: str, start: float, end: float):
    model = get_whisper_model()
    segments, _ = model.transcribe(
        path,
        beam_size=settings.WHISPER_BEAM_SIZE,
        temperature=None,
        return_segments=True,
        split_on_word=False,
        word_timestamps=False,
    )
    text = " ".join(
        seg.text
        for seg in segments
        if seg.start >= start and seg.end <= end
    )
    return {"start": start, "end": end, "text": text.strip()}