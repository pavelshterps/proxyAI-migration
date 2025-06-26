import os
from celery_app import celery_app
from config.settings import settings
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Lazy singletons
_whisper_model = None
_diarizer = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=0,
            intra_threads=1,
            inter_threads=1,
        )
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir="/hf_cache",
        )
    return _diarizer

@celery_app.task(name="tasks.diarize_full")
def diarize_full(path: str):
    diarizer = get_diarizer()
    timeline = diarizer({"audio": path})
    # split into speaker‐labelled WAV and dispatch each to transcribe_segments
    segments = []
    for turn, _, speaker in timeline.itertracks(yield_label=True):
        out = f"{path}.{speaker}.{turn.start:.2f}-{turn.end:.2f}.wav"
        os.system(f"ffmpeg -i {path} -ss {turn.start} -to {turn.end} -c copy {out}")
        segments.append((out, speaker))
    res = []
    for out, speaker in segments:
        res.extend(transcribe_segments.delay(out).get())
    return res

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(path: str):
    model = get_whisper()
    segments, info = model.transcribe(path, beam_size=settings.ALIGN_BEAM_SIZE)
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]