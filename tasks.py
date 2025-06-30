import os
import json
import logging
from pathlib import Path

import structlog
import webrtcvad
from pydub import AudioSegment
from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from prometheus_client import Counter

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

logger = structlog.get_logger()
MODEL_LOAD_ERRORS = Counter("proxyai_model_load_errors_total", "Failures loading models")
TRANSCRIBE_ERRORS = Counter("proxyai_transcribe_errors_total", "Failures during transcription")

# singletons so we only load once per worker
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = os.getenv("WHISPER_MODEL_PATH", "/hf_cache/models--guillaumekln--faster-whisper-medium")
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        try:
            logger.info("Loading WhisperModel", path=model_path, device=device, compute=compute)
            _whisper_model = WhisperModel(model_path, device=device, compute_type=compute)
            logger.info("WhisperModel loaded")
        except Exception as e:
            MODEL_LOAD_ERRORS.inc()
            logger.error("Failed to load WhisperModel", path=model_path, error=str(e))
            raise
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            logger.info("Loading pyannote Pipeline", cache_dir=cache_dir)
            _diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization", cache_dir=cache_dir)
            logger.info("Diarizer loaded")
        except Exception as e:
            MODEL_LOAD_ERRORS.inc()
            logger.error("Failed to load pyannote Pipeline", error=str(e))
            raise
    return _diarizer


def split_audio_vad(audio_path: Path, aggressiveness: int = 3):
    """Use webrtcvad to split speech regions."""
    audio = AudioSegment.from_file(str(audio_path)).set_channels(1).set_frame_rate(16000)
    vad = webrtcvad.Vad(aggressiveness)
    sample_rate = 16000
    frame_duration_ms = 30
    frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2
    raw = audio.raw_data
    frames = [raw[i : i + frame_bytes] for i in range(0, len(raw), frame_bytes)]

    segments = []
    current_start = None
    for i, frame in enumerate(frames):
        is_speech = False
        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception:
            pass
        t = i * frame_duration_ms / 1000.0
        if is_speech and current_start is None:
            current_start = t
        elif not is_speech and current_start is not None:
            segments.append((current_start, t))
            current_start = None
    if current_start is not None:
        segments.append((current_start, len(audio) / 1000.0))
    return segments or [(0.0, None)]


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting transcription", src=str(src))
    # 1) Try VAD‐based segments, else fixed windows
    segments = split_audio_vad(src)
    logger.info("Using segments count", count=len(segments))

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug("Transcribing segment", idx=idx, start=start, end=end)
        try:
            result = whisper.transcribe(
                str(src),
                beam_size=5,
                language="ru",
                vad_filter=True,
                word_timestamps=True,
                offset=start,
                duration=(None if end is None else end - start),
            )
            text = result["segments"][0]["text"]
        except Exception as e:
            TRANSCRIBE_ERRORS.inc()
            logger.error("Segment transcription failed", idx=idx, error=str(e))
            text = ""
        transcript.append({"segment": idx, "start": start, "end": end, "text": text})

    out_path = dst_dir / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Transcription complete", out_path=str(out_path))


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting diarization", src=str(src))
    try:
        diarization = diarizer(str(src))
    except Exception as e:
        logger.error("Diarization failed", error=str(e))
        return

    speakers = []
    for turn, _, spk in diarization.itertracks(yield_label=True):
        speakers.append({"start": turn.start, "end": turn.end, "speaker": spk})

    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Diarization complete", out_path=str(out_path))