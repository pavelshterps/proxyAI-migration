import os
import json
import logging
from pathlib import Path

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from celery_app import celery_app as app
from config.settings import settings

logger = logging.getLogger(__name__)

# Singletons: model & diarizer live once per worker process
_whisper_model: WhisperModel | None = None
_diarizer: Pipeline | None = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading WhisperModel once at startup: "
            f"{{'path': '{settings.WHISPER_MODEL_PATH}', "
            f"'device': '{settings.WHISPER_DEVICE}', "
            f"'compute_type': '{settings.WHISPER_COMPUTE_TYPE}'}}"
        )
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            device_index=settings.WHISPER_DEVICE_INDEX,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            num_threads=settings.WHISPER_INTER_THREADS,
        )
        logger.info("WhisperModel loaded (quantized CTranslate2 format)")
    return _whisper_model


def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache at '{cache_dir}'…")
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            cache_dir=cache_dir,
        )
        logger.info("Diarizer loaded")
    return _diarizer


@app.task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    """
    Transcribe the audio file via WhisperModel.
    Uses faster_whisper's vad_filter to break into spoken segments,
    then writes out transcript.json with per-segment text & timings.
    """
    whisper = get_whisper_model()

    audio_path = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    output_dir = Path(settings.RESULTS_FOLDER) / upload_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for upload '{upload_id}': {audio_path}")

    # run transcription with built-in VAD segmentation
    result = whisper.transcribe(
        str(audio_path),
        beam_size=settings.WHISPER_BEAM_SIZE,
        language=settings.WHISPER_LANGUAGE,
        vad_filter=True,
        word_timestamps=True,
    )

    transcript_segments = []
    for seg in result["segments"]:
        transcript_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })

    out_path = output_dir / "transcript.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript_segments, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Transcription complete for upload '{upload_id}', "
        f"{len(transcript_segments)} segments → {out_path}"
    )


@app.task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    """
    Run full‐file speaker diarization with pyannote.
    Writes out diarization.json with [{start,end,speaker},…].
    """
    diarizer = get_diarizer()

    audio_path = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    output_dir = Path(settings.RESULTS_FOLDER) / upload_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for upload '{upload_id}': {audio_path}")

    diarization = diarizer(str(audio_path))

    diarization_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    out_path = output_dir / "diarization.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Diarization complete for upload '{upload_id}', "
        f"{len(diarization_segments)} turns → {out_path}"
    )