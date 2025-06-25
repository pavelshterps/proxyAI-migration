import os
from celery_app import app
import config.settings as settings
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# Singletons so we only load each big model once per worker process
_diarization_pipeline = None
_whisper_model = None

def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        _diarization_pipeline = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _diarization_pipeline

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=0,
            inter_threads=1,
            intra_threads=1,
        )
    return _whisper_model

@app.task(name="tasks.diarize_full", queue="preprocess_cpu")
def diarize_full(filepath: str):
    """
    Run pyannote speaker‐diarization on the entire file.
    """
    pipeline = get_diarization_pipeline()
    return pipeline({"audio": filepath})

@app.task(name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(diarization_result, filepath: str):
    """
    For each segment from diarization, run Whisper only on that slice.
    """
    model = get_whisper_model()
    results = []
    timeline = diarization_result.get_timeline()
    for segment in timeline.segments:
        start, end = segment.start, segment.end
        # faster-whisper returns {"segments": [...]}
        transcription = model.transcribe(
            filepath,
            beam_size=settings.ALIGN_BEAM_SIZE,
            start=start,
            end=end
        )
        text = " ".join(s.text for s in transcription["segments"])
        results.append({
            "start": start,
            "end": end,
            "text": text
        })
    return results

@app.task(name="tasks.transcribe_full", queue="preprocess_gpu")
def transcribe_full(filepath: str):
    """
    Chunked Whisper on the whole file (if you ever want a single-pass fallback).
    """
    model = get_whisper_model()
    transcription = model.transcribe(
        filepath,
        beam_size=settings.ALIGN_BEAM_SIZE
    )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in transcription["segments"]
    ]