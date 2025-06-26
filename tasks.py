# tasks.py

import os
from typing import List, Dict

from celery_app import celery_app
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings

# Global, lazy-loaded models
_whisper_model: WhisperModel = None
_diarizer_pipeline: Pipeline = None

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            device=settings.WHISPER_DEVICE,
            compute_type="float16",
            device_index=0,
            inter_threads=1,
            intra_threads=1
        )
    return _whisper_model

def get_diarizer_pipeline() -> Pipeline:
    global _diarizer_pipeline
    if _diarizer_pipeline is None:
        # Load without device argument
        _diarizer_pipeline = Pipeline.from_pretrained(
            settings.PYANNOTE_MODEL,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        # Move pipeline to the configured device
        _diarizer_pipeline.to(settings.WHISPER_DEVICE)
    return _diarizer_pipeline

@celery_app.task(name="tasks.diarize_full")
def diarize_full(filepath: str) -> List[Dict]:
    """
    Run speaker diarization in chunks and then enqueue transcription.
    Returns list of {'start', 'end', 'speaker'} dicts.
    """
    pipeline = get_diarizer_pipeline()
    chunk_len = settings.DIARIZE_CHUNK_LENGTH  # e.g. 30 seconds

    # run diarization over entire file in sliding chunks
    all_segments = []
    audio_duration = pipeline.get_timeline(filepath).extent.duration  # uses pyannote's helper
    for offset in range(0, int(audio_duration), chunk_len):
        segment = {"uri": filepath, "start": offset, "end": offset + chunk_len}
        diarization = pipeline({"audio": filepath, **segment})
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            all_segments.append({
                "start": turn.start + offset,
                "end": turn.end + offset,
                "speaker": speaker
            })

    # Enqueue transcription of each segment
    transcribe_segments.delay(filepath, all_segments, settings.TUSD_ENDPOINT)
    return all_segments

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(
    filepath: str,
    segments: List[Dict],
    tusd_endpoint: str
) -> List[Dict]:
    """
    Transcribe each diarized segment with Whisper.
    Returns list of {'start','end','speaker','text'}.
    """
    model = get_whisper_model()
    results = []

    for seg in segments:
        start, end = seg["start"], seg["end"]
        # WhisperModel returns (transcription, logprobs)
        transcription, _ = model.transcribe(
            filepath,
            start=start,
            end=end,
            beam_size=settings.ALIGN_BEAM_SIZE,
            model=settings.ALIGN_MODEL_NAME,
            snippet_format=settings.SNIPPET_FORMAT
        )
        results.append({
            "start": start,
            "end": end,
            "speaker": seg["speaker"],
            "text": transcription
        })

    # Optionally upload to TUSD or store results here...
    # e.g. requests.post(f"{tusd_endpoint}/files/", json=results)
    return results