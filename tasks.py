import os
import json
import logging
from pathlib import Path
from celery import shared_task
from faster_whisper import WhisperModel, WhisperError
from pyannote.audio import Pipeline, PyannoteError
from pydub import AudioSegment

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER, settings

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device = settings.WHISPER_DEVICE
        compute = settings.WHISPER_COMPUTE_TYPE
        logger.info(f"Loading WhisperModel {{path={model_path}, device={device}, compute={compute}}}")
        _whisper_model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute,
            device_index=settings.WHISPER_DEVICE_INDEX,
        )
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache {cache_dir}")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=str(cache_dir),
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("Diarizer loaded")
    return _diarizer

def split_audio_fixed_windows(audio_path: Path):
    window_s = settings.SEGMENT_LENGTH_S
    audio = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    return [
        (start / 1000.0, min(start + window_ms, length_ms) / 1000.0)
        for start in range(0, length_ms, window_ms)
    ]

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcription started: {src}")
    segments = split_audio_fixed_windows(src)
    transcript = []

    for idx, (start, end) in enumerate(segments):
        try:
            res = whisper.transcribe(
                str(src),
                beam_size=settings.WHISPER_BEAM_SIZE,
                language="ru",
                vad_filter=True,
                word_timestamps=True,
                offset=start,
                duration=end - start,
            )
            text = res["segments"][0]["text"]
        except WhisperError as e:
            logger.exception(f"Whisper error in segment {idx}: {e}")
            text = ""
        transcript.append({"segment": idx, "start": start, "end": end, "text": text})

    out_path = dst_dir / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Transcription saved: {out_path}")

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Diarization started: {src}")
    try:
        dia = diarizer(str(src))
        speakers = [
            {"start": turn.start, "end": turn.end, "speaker": spk}
            for turn, _, spk in dia.itertracks(yield_label=True)
        ]
    except PyannoteError as e:
        logger.exception(f"Diarizer error: {e}")
        speakers = []

    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Diarization saved: {out_path}")