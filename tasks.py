import os
import logging
import huggingface_hub
# Monkey-patch HF snapshot_download to allow local quantized model paths
_original_snapshot_download = huggingface_hub.snapshot_download
def _snapshot_download_override(repo_id, *args, **kwargs):
    if os.path.exists(repo_id):
        return repo_id
    return _original_snapshot_download(repo_id, *args, **kwargs)
huggingface_hub.snapshot_download = _snapshot_download_override
from faster_whisper import WhisperModel
import faster_whisper.utils as fw_utils
import faster_whisper.transcribe as fw_transcribe

import huggingface_hub
from huggingface_hub import snapshot_download

import huggingface_hub.utils._validators as hf_validators
# Monkey-patch validate_repo_id to accept local filesystem paths
_orig_validate_repo_id = hf_validators.validate_repo_id
def _validate_repo_id_override(repo_id):
    if os.path.exists(repo_id):
        return
    return _orig_validate_repo_id(repo_id)
hf_validators.validate_repo_id = _validate_repo_id_override

# Monkey-patch download_model to accept local quantized model directories
_original_download_model = fw_utils.download_model
def _download_model_override(repo_id, *args, **kwargs):
    if os.path.exists(repo_id):
        return repo_id
    return _original_download_model(repo_id, *args, **kwargs)
fw_utils.download_model = _download_model_override
fw_transcribe.download_model = _download_model_override

from pyannote.audio import Pipeline
from celery import Task
from celery_app import celery_app
from config.settings import settings

logger = logging.getLogger(__name__)

_whisper_model: WhisperModel | None = None
_diarizer: Pipeline | None = None

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        params = {
            "model_size_or_path": "/hf_cache/quantized/whisper-medium-float16",
            "device": settings.WHISPER_DEVICE,
            "compute_type": settings.WHISPER_COMPUTE_TYPE,
            "device_index": settings.WHISPER_DEVICE_INDEX,
        }
        logger.info(f"Loading WhisperModel with params: {params}")
        _whisper_model = WhisperModel(**params)
        logger.info("WhisperModel loaded (quantized CTranslate2 format)")
    return _whisper_model

def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_MODEL,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.HF_CACHE_DIR,
        )
    return _diarizer

@celery_app.task(
    name="tasks.diarize_full",
    bind=True,
    acks_late=True,
    ignore_result=False,
)
def diarize_full(self: Task, wav_path: str) -> list[dict]:
    logger.info(f"Starting diarize_full on {wav_path}")
    diarizer = get_diarizer()
    timeline = diarizer(wav_path)
    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in timeline.itertracks(yield_label=True)
    ]

    logger.info(f"Scheduling transcribe_segments for {wav_path}")
    transcribe_segments.apply_async(
        args=(wav_path,),
        queue="preprocess_gpu"
    )

    return segments

@celery_app.task(
    name="tasks.transcribe_segments",
    bind=True,
    acks_late=True,
    ignore_result=False,
)
def transcribe_segments(self: Task, wav_path: str) -> list[dict]:
    logger.info(f"Starting transcribe_segments on {wav_path}")
    model = get_whisper_model()
    segments, _info = model.transcribe(
        wav_path,
        beam_size=settings.WHISPER_BEAM_SIZE,
        best_of=settings.WHISPER_BEST_OF,
        task=settings.WHISPER_TASK,
    )

    out = []
    for start, end, text in segments:
        out.append({
            "start": float(start),
            "end":   float(end),
            "text":  text.strip(),
        })

    if settings.CLEAN_UP_UPLOADS:
        try:
            os.remove(wav_path)
        except Exception:
            logger.warning(f"Failed to remove {wav_path}", exc_info=True)

    return out