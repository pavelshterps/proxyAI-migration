import os
from celery import Celery
from config.settings import settings
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

# Celery app setup
celery_app = Celery(
    'proxyai',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Task routing
celery_app.conf.task_routes = {
    "tasks.diarize_full": {"queue": "preprocess_cpu"},
    "tasks.transcribe_segments": {"queue": "preprocess_gpu"},
}

# Lazy-loaded diarizer pipeline
_diarizer_pipeline = None


def get_diarizer_pipeline():
    global _diarizer_pipeline
    if _diarizer_pipeline is None:
        # Download & cache the model on first build (at runtime)
        _diarizer_pipeline = Pipeline.from_pretrained(
            settings.PYANNOTE_MODEL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
    # Move pipeline to the configured device
    device = torch.device(settings.WHISPER_DEVICE)
    _diarizer_pipeline.to(device)
    return _diarizer_pipeline


@celery_app.task(name="tasks.diarize_full")
def diarize_full(file_path: str):
    """
    1. Run speaker diarization to split audio into per-speaker segments.
    2. Hand off each segment to Whisper in GPU.
    3. Combine and return the list of (start, end, speaker, text).
    """
    pipeline = get_diarizer_pipeline()
    # inference happens on GPU (float16)
    diarization = pipeline({"audio": file_path})
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    # hand off to transcription
    transcribe_segments.delay(file_path, segments)
    return segments


@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(file_path: str, segments: list):
    """
    Transcribe each segment with WhisperModel on GPU.
    """
    model = WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        device_index=settings.WHISPER_DEVICE_INDEX,
        inter_threads=1,
        intra_threads=1,
    )
    results = []
    for start, end, speaker in segments:
        duration = end - start
        segments_gen, info = model.transcribe(
            file_path,
            beam_size=settings.WHISPER_BEAM_SIZE,
            segment_first=start,
            segment_last=end,
            vad_filter=False,
        )
        # take joined text
        text = "".join([segment.text for segment in segments_gen])
        results.append((start, end, speaker, text.strip()))
    # store in backend under job id
    celery_app.backend.store_result(
        task_id=os.path.splitext(os.path.basename(file_path))[0],
        result=results,
        status='SUCCESS',
    )
    return results