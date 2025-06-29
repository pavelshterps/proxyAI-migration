import os
import json
import logging
import wave
from pathlib import Path

import webrtcvad
from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            device_index=settings.WHISPER_DEVICE_INDEX,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache, exist_ok=True)
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            cache_dir=cache
        )
        logger.info("Diarizer loaded")
    return _diarizer


def vad_segment_generator(
    audio_path: str,
    sample_rate: int = 16000,
    frame_duration_ms: int = 30,
    padding_duration_ms: int = 300,
    aggressiveness: int = 3,
):
    vad = webrtcvad.Vad(aggressiveness)
    wf = wave.open(audio_path, "rb")
    assert wf.getnchannels() == 1
    assert wf.getsampwidth() == 2
    assert wf.getframerate() == sample_rate

    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
    segments = []
    timestamp = 0.0
    current_start = None
    silence_frames = 0
    padding_frames = padding_duration_ms // frame_duration_ms

    data = wf.readframes(frame_size // 2)
    while len(data) == frame_size:
        speech = vad.is_speech(data, sample_rate)
        if speech:
            if current_start is None:
                current_start = timestamp
            silence_frames = 0
        else:
            if current_start is not None:
                silence_frames += 1
                if silence_frames >= padding_frames:
                    segments.append((current_start, timestamp))
                    current_start = None
                    silence_frames = 0
        timestamp += frame_duration_ms / 1000.0
        data = wf.readframes(frame_size // 2)

    if current_start is not None:
        segments.append((current_start, timestamp))
    wf.close()

    return segments or [(0.0, None)]


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    in_f = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {in_f}; running VAD…")
    segments = vad_segment_generator(str(in_f))
    logger.info(f"VAD segments: {len(segments)}")

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"Segment {idx}: {start}-{end}")
        res = whisper.transcribe(
            str(in_f),
            beam_size=settings.WHISPER_BEAM_SIZE,
            language="ru",
            offset=start,
            duration=None if end is None else (end - start),
            vad_filter=False,
            word_timestamps=True,
        )
        text = res["segments"][0]["text"]
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text,
        })

    with open(out_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    logger.info(f"Transcript saved to {out_dir/'transcript.json'}")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    in_f = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Diarizing {in_f}…")
    diarization = diarizer(str(in_f))

    segments = []
    for turn, _, spk in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end":   turn.end,
            "speaker": spk,
        })

    with open(out_dir / "diarization.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Diarization saved to {out_dir/'diarization.json'}")