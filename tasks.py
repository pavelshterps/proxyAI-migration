from celery_app import celery_app
from config.settings import settings
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

_diarizer = None
_model = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            device="cpu"
        )
    return _diarizer

def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            device_index=0,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            inter_threads=settings.INTER_THREADS,
            intra_threads=settings.INTRA_THREADS,
            cache_dir="/hf_cache"
        )
    return _model

@celery_app.task(name="tasks.diarize_full", queue="preprocess_cpu")
def diarize_full(filepath: str) -> list[dict]:
    diarizer = get_diarizer()
    diarization = diarizer(filepath)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return segments

@celery_app.task(name="tasks.transcribe_full", queue="preprocess_gpu")
def transcribe_full(filepath: str, segments: list[dict]) -> dict:
    model = get_model()
    results = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        text_segs, _ = model.transcribe(
            filepath,
            beam_size=settings.ALIGN_BEAM_SIZE,
            segment=[start, end],
        )
        text = "".join([t.text for t in text_segs])
        results.append({
            "start": start,
            "end": end,
            "speaker": seg["speaker"],
            "text": text,
        })
    return {"segments": results}