import os
from celery_app import app
from config.settings import settings

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ---- Lazy singletons ----
_model: WhisperModel | None = None
_diarizer: Pipeline | None = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.device,
            compute_type=settings.whisper_compute_type,
            device_index=0,
            intra_threads=1,
            inter_threads=1
        )
    return _model

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            settings.pyannote_protocol,
            use_auth_token=settings.hf_token or settings.huggingface_token
        )
    return _diarizer

# ---- GPU task: transcribe a single 30s chunk ----
@app.task(name="tasks.transcribe_segments")
def transcribe_segments(path: str) -> list[dict]:
    model = get_model()
    segments, info = model.transcribe(
        path,
        beam_size=settings.align_beam_size,
        word_timestamps=True
    )
    # Flatten results into serializable dicts
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]

# ---- CPU task: diarize full file & fan-out to GPU ----
@app.task(name="tasks.diarize_full")
def diarize_full(path: str) -> list[dict]:
    diarizer = get_diarizer()
    diarization = diarizer({"audio": path})
    results: list[dict] = []

    # For each speaker segment, slice out the audio snippet and send to GPU
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        snippet = f"{path}_{turn.start:.2f}_{turn.end:.2f}.wav"
        # Use ffmpeg/tusd or local trim to create snippet
        # ... (left as your existing code)
        # Now send snippet for transcription
        segs = transcribe_segments.delay(snippet).get()
        for s in segs:
            s["speaker"] = speaker
            results.append(s)

    return results