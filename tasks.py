import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

logger = logging.getLogger(__name__)

# Singletons so we only load each model once per worker process
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = os.getenv(
            "WHISPER_MODEL_PATH",
            "/hf_cache/models--guillaumekln--faster-whisper-medium"
        )
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        logger.info(
            f"Loading WhisperModel {{'path': '{model_path}', 'device': '{device}', 'compute': '{compute}'}}"
        )
        _whisper_model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute
        )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache '{cache_dir}'")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=cache_dir
        )
        logger.info("Diarizer loaded")
    return _diarizer


def split_audio_fixed_windows(audio_path: Path):
    """
    Split the audio into fixed-length windows (default 30s) for batching.
    Returns a list of (start_sec, end_sec) tuples.
    """
    window_s = int(os.getenv("SEGMENT_LENGTH_S", "30"))
    audio = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    segments = []
    for start_ms in range(0, length_ms, window_ms):
        end_ms = min(start_ms + window_ms, length_ms)
        segments.append((start_ms / 1000.0, end_ms / 1000.0))
    return segments


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    """
    1) Splits audio into fixed‐window segments
    2) Transcribes each with Whisper
    3) Dumps RESULTS_FOLDER/<upload_id>/transcript.json
    """
    whisper = get_whisper_model()
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for '{src}'")
    segments = split_audio_fixed_windows(src)
    logger.info(f"  -> {len(segments)} segments of up to {os.getenv('SEGMENT_LENGTH_S','30')}s")

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"  Transcribing segment {idx}: {start:.1f}s → {end:.1f}s")
        result = whisper.transcribe(
            str(src),
            beam_size=5,
            language="ru",
            vad_filter=True,
            word_timestamps=True,
            offset=start,
            duration=(end - start),
        )
        text = result["segments"][0]["text"]
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text
        })

    out_path = dst_dir / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Transcription complete: saved to '{out_path}'")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    """
    1) Runs speaker diarization on the whole file
    2) Dumps RESULTS_FOLDER/<upload_id>/diarization.json
    """
    diarizer = get_diarizer()
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for '{src}'")
    diarization = diarizer(str(src))

    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Diarization complete: saved to '{out_path}'")