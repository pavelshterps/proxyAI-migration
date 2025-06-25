import os
import tempfile
from celery_app import app
from config.settings import (
    DEVICE,
    WHISPER_MODEL,
    WHISPER_COMPUTE_TYPE,
    ALIGN_BEAM_SIZE,
    PYANNOTE_PROTOCOL,
    HUGGINGFACE_TOKEN
)
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

_diarizer = None
_transcriber = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN,
            cache_dir=os.getenv("HF_HOME")
        )
    return _diarizer

def get_transcriber():
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperModel(
            model_size_or_path=WHISPER_MODEL,
            device=DEVICE,
            device_index=0,
            compute_type=WHISPER_COMPUTE_TYPE,
            inter_threads=1,
            intra_threads=1,
            cache_dir=os.getenv("HF_HOME")
        )
    return _transcriber

@app.task(name="tasks.diarize_full")
def diarize_full(path: str):
    pipeline = get_diarizer()
    output = pipeline(path)
    segments = []
    for turn in output.get_timeline().support():
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": turn.label
        })
    # enqueue transcription stage
    app.send_task(
        "tasks.transcribe_segments",
        args=[path, segments],
        queue="preprocess_gpu",
        task_id=diarize_full.request.id
    )
    return segments

@app.task(name="tasks.transcribe_segments")
def transcribe_segments(path: str, segments: list):
    transcriber = get_transcriber()
    results = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        # cut chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        os.system(f"ffmpeg -y -i {path} -ss {start} -to {end} {tmp_path}")
        # transcribe chunk
        txt_segs, _ = transcriber.transcribe(
            tmp_path,
            beam_size=ALIGN_BEAM_SIZE
        )
        text = " ".join([s.text for s in txt_segs])
        results.append({**seg, "text": text})
        os.remove(tmp_path)
    return results