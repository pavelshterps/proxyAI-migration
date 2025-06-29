import os
import json
import logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

from config.settings import settings

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        path = settings.WHISPER_MODEL_PATH
        device = settings.WHISPER_DEVICE
        compute = settings.WHISPER_COMPUTE_TYPE
        logger.info(f"Loading WhisperModel {{'path': '{path}', 'device': '{device}', 'compute': '{compute}'}}")
        _whisper_model = WhisperModel(
            path,
            device=device,
            compute_type=compute
        )
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading pyannote Pipeline into cache '{cache_dir}'")
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            cache_dir=cache_dir
        )
        logger.info("Diarizer loaded")
    return _diarizer

def split_audio_fixed_windows(audio_path: Path):
    window_s = settings.SEGMENT_LENGTH_S
    audio = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    segments = []
    for start_ms in range(0, length_ms, window_ms):
        end_ms = min(start_ms + window_ms, length_ms)
        segments.append((start_ms/1000.0, end_ms/1000.0))
    return segments

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst = Path(settings.RESULTS_FOLDER) / upload_id
    dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for '{src}'")
    segments = split_audio_fixed_windows(src)
    logger.info(f"  → {len(segments)} segments (up to {settings.SEGMENT_LENGTH_S}s)")

    transcript = []
    for i, (start, end) in enumerate(segments):
        logger.debug(f"  Transcribing segment {i}: {start:.1f}s–{end:.1f}s")
        res = whisper.transcribe(
            str(src),
            beam_size=settings.WHISPER_BEAM_SIZE,
            language="ru",
            vad_filter=True,
            word_timestamps=True,
            offset=start,
            duration=end - start,
        )
        txt = res["segments"][0]["text"].strip()
        transcript.append({"segment": i, "start": start, "end": end, "text": txt})

    out = dst / "transcript.json"
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Transcription complete → '{out}'")

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst = Path(settings.RESULTS_FOLDER) / upload_id
    dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for '{src}'")
    dia = diarizer(str(src))

    speakers = []
    for turn, _, spk in dia.itertracks(yield_label=True):
        speakers.append({"start": turn.start, "end": turn.end, "speaker": spk})

    out = dst / "diarization.json"
    out.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Diarization complete → '{out}'")