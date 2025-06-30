import os
import json
import logging
from pathlib import Path
from typing import List, Tuple

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import webrtcvad

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        model_path = os.getenv(
            "WHISPER_MODEL_PATH",
            "/hf_cache/models--guillaumekln--faster-whisper-medium"
        )
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        try:
            logger.info(
                "Loading WhisperModel",
                extra={"model_path": model_path, "device": device, "compute": compute},
            )
            _whisper_model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute
            )
            logger.info("WhisperModel loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load WhisperModel from {model_path}: {e}")
            raise
    return _whisper_model


def get_diarizer() -> Pipeline:
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            logger.info("Loading pyannote Pipeline", extra={"cache_dir": cache_dir})
            _diarizer = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                cache_dir=cache_dir
            )
            logger.info("Diarizer loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load diarizer into {cache_dir}: {e}")
            raise
    return _diarizer


def frame_generator(frame_duration_ms: int,
                    audio_bytes: bytes,
                    sample_rate: int) -> List[bytes]:
    """Yield successive frames of audio_bytes for VAD."""
    n_bytes_per_frame = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit mono
    offset = 0
    frames = []
    while offset + n_bytes_per_frame < len(audio_bytes):
        frames.append(audio_bytes[offset:offset + n_bytes_per_frame])
        offset += n_bytes_per_frame
    return frames


def vad_collector(sample_rate: int,
                  frame_duration_ms: int,
                  padding_duration_ms: int,
                  vad: webrtcvad.Vad,
                  frames: List[bytes]) -> List[Tuple[float, float]]:
    """
    Returns time windows (in sec) where speech is detected.
    """
    num_padding_frames = padding_duration_ms // frame_duration_ms
    speech_windows = []
    ring_buffer = []
    triggered = False
    window_start = 0.0

    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append((i, is_speech))
            if sum(1 for _, speech in ring_buffer if speech) > 0.9 * len(ring_buffer):
                triggered = True
                window_start = ring_buffer[0][0] * frame_duration_ms / 1000.0
                ring_buffer.clear()
        else:
            ring_buffer.append((i, is_speech))
            if sum(1 for _, speech in ring_buffer if not speech) > 0.9 * len(ring_buffer):
                window_end = i * frame_duration_ms / 1000.0
                speech_windows.append((window_start, window_end))
                triggered = False
                ring_buffer.clear()

    # Catch last segment
    if triggered:
        speech_windows.append((window_start, len(frames) * frame_duration_ms / 1000.0))

    return speech_windows


def split_audio_vad(audio_path: Path) -> List[Tuple[float, float]]:
    """
    Perform VAD segmentation and return list of (start_s, end_s) for speech.
    """
    # Load, convert to mono 16kHz
    audio = AudioSegment.from_file(str(audio_path))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    pcm = audio.raw_data

    vad_mode = int(os.getenv("VAD_MODE", "2"))
    vad = webrtcvad.Vad(vad_mode)

    frame_ms = int(os.getenv("VAD_FRAME_MS", "30"))
    padding_ms = int(os.getenv("VAD_PADDING_MS", "300"))

    frames = frame_generator(frame_ms, pcm, 16000)
    speech_segments = vad_collector(16000, frame_ms, padding_ms, vad, frames)

    logger.info(f"VAD found {len(speech_segments)} speech segment(s)")
    return speech_segments


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for '{src}'")
    segments = split_audio_vad(src)
    if not segments:
        # fallback to whole file
        segments = [(0.0, None)]
        logger.warning("No VAD segments detected; falling back to full-file transcription")

    transcript = []
    for idx, (start, end) in enumerate(segments):
        try:
            logger.debug(f"Transcribing segment {idx}: {start:.2f}s → {end}")
            result = whisper.transcribe(
                str(src),
                beam_size=5,
                language="ru",
                vad_filter=False,            # we've already segmented
                word_timestamps=True,
                offset=start,
                duration=None if end is None else (end - start),
            )
            text = result["segments"][0]["text"]
        except Exception as e:
            logger.exception(f"Error transcribing segment {idx}; skipping: {e}")
            text = ""
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
    diarizer = get_diarizer()
    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for '{src}'")
    try:
        diarization = diarizer(str(src))
    except Exception as e:
        logger.exception(f"Diarization pipeline failed on '{src}': {e}")
        diarization = []

    speakers = []
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
    else:
        logger.warning("No diarization output; saving empty speaker list")

    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Diarization complete: saved to '{out_path}'")