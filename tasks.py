# tasks.py

import os
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from celery import shared_task
from config.settings import settings

# Preload diarizer at import time (caches under HF_HOME)
diarizer = Pipeline.from_pretrained(
    settings.PYANNOTE_MODEL,
    use_auth_token=settings.HF_TOKEN,
)

@shared_task(name="tasks.diarize_full")
def diarize_full(filepath: str):
    """Run full-file speaker diarization on CPU."""
    # returns list of (start, end, speaker_id)
    return list(diarizer(filepath).itertracks())

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(filepath: str, segments: list):
    """Chunked Whisper transcription on GPU."""
    model = WhisperModel(
        settings.WHISPER_MODEL,
        device="cuda",
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        device_index=0,
        intra_threads=1,
        inter_threads=1,
    )

    results = []
    for start, end, speaker in segments:
        # load slice, transcribe, collect text per segment
        segment = model.transcribe(filepath, segment=[start, end])
        results.append({"start": start, "end": end, "speaker": speaker, "text": segment.text})
    return results