import os
import json
import wave
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import webrtcvad
from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Singleton instances (one per worker process)
# -----------------------------------------------------------------------------
_whisper_model: Optional[WhisperModel] = None
_diarizer:         Optional[Pipeline]     = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        model_path   = settings.WHISPER_MODEL_PATH
        device       = settings.WHISPER_DEVICE
        compute_type = settings.WHISPER_COMPUTE_TYPE
        beam_size    = settings.WHISPER_BEAM_SIZE

        logger.info(
            f"Loading WhisperModel once at startup: "
            f"{{'model_path':'{model_path}',"
            f"'device':'{device}',"
            f"'compute_type':'{compute_type}',"
            f"'beam_size':{beam_size}}}"
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
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------
# VAD-based segmentation using webrtcvad
# --------------------------------------
def _read_wave(path: str) -> Tuple[bytes, int]:
    """
    Reads a .wav file and ensures it's 16-bit mono PCM.
    """
    with wave.open(path, "rb") as wf:
        num_channels = wf.getnchannels()
        samp_width   = wf.getsampwidth()
        rate         = wf.getframerate()
        assert num_channels == 1, "Only mono WAV supported"
        assert samp_width == 2,     "Only 16-bit WAV supported"
        assert rate in (8000, 16000, 32000, 48000), "Unsupported sampling rate"
        frames = wf.readframes(wf.getnframes())
    return frames, rate


def _frame_generator(
    frame_duration_ms: int, audio: bytes, sample_rate: int
):
    """
    Yield successive audio frames of length frame_duration_ms.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # bytes per frame
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    while offset + n <= len(audio):
        yield audio[offset:offset + n], timestamp
        timestamp += duration
        offset += n


def _vad_collector(
    vad: webrtcvad.Vad,
    frames,
    sample_rate: int,
    padding_duration_ms: int = 300
) -> List[Tuple[float, float]]:
    """
    Collect voiced segments from the frame generator.
    Returns a list of (segment_start, segment_end) in seconds.
    """
    num_padding_frames = int(padding_duration_ms / 30)
    ring_buffer = []
    voiced_segments = []

    triggered = False
    segment_start = 0.0

    for frame, ts in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        ring_buffer.append((frame, ts, is_speech))

        if len(ring_buffer) > num_padding_frames:
            ring_buffer.pop(0)

        if not triggered:
            # start when enough speech in buffer
            num_voiced = sum(1 for f in ring_buffer if f[2])
            if num_voiced > 0.9 * len(ring_buffer):
                triggered = True
                segment_start = ring_buffer[0][1]
                ring_buffer.clear()
        else:
            # end when enough silence
            num_unvoiced = sum(1 for f in ring_buffer if not f[2])
            if num_unvoiced > 0.9 * len(ring_buffer):
                segment_end = ts + 0.03  # end of last frame
                voiced_segments.append((segment_start, segment_end))
                triggered = False
                ring_buffer.clear()

    # if we're still in a segment when done
    if triggered:
        last_ts = ring_buffer[-1][1] + 0.03
        voiced_segments.append((segment_start, last_ts))

    return voiced_segments


def _split_segments(audio_path: str) -> List[Tuple[float, Optional[float]]]:
    """
    Split the full WAV into voice-active segments using webrtcvad.
    """
    try:
        aggressiveness = settings.WHISPER_VAD_AGGRESSIVENESS
        vad = webrtcvad.Vad(aggressiveness)

        audio_bytes, rate = _read_wave(audio_path)
        frames = list(_frame_generator(30, audio_bytes, rate))
        segments = _vad_collector(vad, frames, rate, padding_duration_ms=300)

        logger.info(f"VAD split produced {len(segments)} segments")
        return segments or [(0.0, None)]
    except Exception:
        logger.exception("VAD segmentation failed, falling back to full-file segment")
        return [(0.0, None)]


# --------------------------------------
# Celery tasks
# --------------------------------------
@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src_wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not src_wav.is_file():
        logger.error(f"Audio file not found: {src_wav}")
        return

    out_dir = _make_result_dir(upload_id)
    logger.info(f"Starting transcription for {upload_id}: {src_wav}")

    segments = _split_segments(str(src_wav))
    transcript = []

    for idx, (start, end) in enumerate(segments):
        logger.debug(f"Segment {idx}: {start}s to {end or 'EOS'}")
        result = whisper.transcribe(
            str(src_wav),
            beam_size=settings.WHISPER_BEAM_SIZE,
            language=settings.WHISPER_LANGUAGE,
            vad_filter=False,            # already split externally
            word_timestamps=True,
            offset=start,
            duration=None if end is None else (end - start),
        )
        text = result["segments"][0]["text"].strip()
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text,
        })

    out_file = out_dir / "transcript.json"
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump(transcript, fp, ensure_ascii=False, indent=2)

    logger.info(f"Transcription complete for {upload_id}, saved to {out_file}")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src_wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    if not src_wav.is_file():
        logger.error(f"Audio file not found: {src_wav}")
        return

    out_dir = _make_result_dir(upload_id)
    logger.info(f"Starting diarization for {upload_id}: {src_wav}")

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

    logger.info(f"Diarization complete for {upload_id}, saved to {out_file}")