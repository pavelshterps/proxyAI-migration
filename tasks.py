import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Singleton instances (one per worker process)
# -----------------------------------------------------------------------------
_whisper_model: Optional[WhisperModel] = None
_diarizer: Optional[Pipeline]       = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        model_path    = settings.WHISPER_MODEL_PATH
        device        = settings.WHISPER_DEVICE
        compute_type  = settings.WHISPER_COMPUTE_TYPE
        beam_size     = settings.WHISPER_BEAM_SIZE

        logger.info(
            f"Loading WhisperModel once at startup: "
            f"{{'model_path': '{model_path}', "
            f"'device':'{device}', 'compute_type':'{compute_type}', 'beam_size':{beam_size}}}"
        )
        _whisper_model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            device_index=settings.WHISPER_DEVICE_INDEX,
        )
        logger.info("WhisperModel loaded (quantized int8)")
    return _whisper_model


def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Loading pyannote Pipeline into cache area: {cache_dir}")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=cache_dir,
        )
        logger.info("Diarizer loaded")
    return _diarizer


def _make_result_dir(upload_id: str) -> Path:
    """Ensure RESULTS_FOLDER/upload_id exists and return it."""
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _split_segments(
    audio_path: str
) -> List[Tuple[float, Optional[float]]]:
    """
    TODO: Replace this stub with zonal-VAD or pyannote-based segmentation for production.
    For now, return the full file as one segment.
    """
    return [(0.0, None)]


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    """
    Split the WAV at UPLOAD_FOLDER/{upload_id}.wav into segments,
    transcribe each with WhisperModel, save JSON to RESULTS_FOLDER/{upload_id}/transcript.json
    """
    whisper = get_whisper_model()

    src_wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not src_wav.is_file():
        logger.error(f"Audio file not found: {src_wav}")
        return

    out_dir = _make_result_dir(upload_id)
    logger.info(f"Starting transcription for {upload_id} → {src_wav}")

    segments = _split_segments(str(src_wav))
    transcript = []

    for idx, (start, end) in enumerate(segments):
        logger.debug(f"Transcribing segment {idx}: {start}s to {end or 'EOS'}")
        result = whisper.transcribe(
            str(src_wav),
            beam_size=settings.WHISPER_BEAM_SIZE,
            language="ru",
            vad_filter=True,
            word_timestamps=True,
            offset=start,
            duration=None if end is None else (end - start),
        )
        text = result["segments"][0]["text"]
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text.strip(),
        })

    out_file = out_dir / "transcript.json"
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump(transcript, fp, ensure_ascii=False, indent=2)

    logger.info(f"Transcription complete for {upload_id}: {out_file}")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    """
    Perform speaker diarization on UPLOAD_FOLDER/{upload_id}.wav,
    save JSON of (start, end, speaker) to RESULTS_FOLDER/{upload_id}/diarization.json
    """
    diarizer = get_diarizer()

    src_wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not src_wav.is_file():
        logger.error(f"Audio file not found: {src_wav}")
        return

    out_dir = _make_result_dir(upload_id)
    logger.info(f"Starting diarization for {upload_id} → {src_wav}")

    # run the full-file diarization
    diarization = diarizer(str(src_wav))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    out_file = out_dir / "diarization.json"
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump(segments, fp, ensure_ascii=False, indent=2)

    logger.info(f"Diarization complete for {upload_id}: {out_file}")