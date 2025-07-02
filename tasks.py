# tasks.py

import os
import json
import logging
from pathlib import Path

from celery import shared_task
from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

from config.settings import settings  # ← импортируем объект настроек

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device = settings.WHISPER_DEVICE.lower()
        compute = settings.WHISPER_COMPUTE_TYPE.lower()

        # На CPU float16 не поддерживается — принудительно int8
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(
                f"Compute type '{compute}' not supported on CPU; falling back to 'int8'"
            )
            compute = "int8"

        logger.info(f"Loading WhisperModel {{'path': '{model_path}', 'device': '{device}', 'compute': '{compute}'}}")
        try:
            # Убраны batch_size и use_auth_token, соответствие текущему API faster-whisper
            _whisper_model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute
            )
        except Exception as e:
            # Ловим любые ошибки (TypeError, ValueError и т.д.)
            logger.warning(
                f"Failed to init WhisperModel with compute_type={compute}: {e}; retrying with 'int8'"
            )
            _whisper_model = WhisperModel(
                model_path,
                device=device,
                compute_type="int8"
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
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("Diarizer loaded")
    return _diarizer


@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    При старте воркера разогреваем модели:
    - diarizer всегда,
    - Whisper только на GPU (чтобы на CPU не пытаться float16).
    """
    device = settings.WHISPER_DEVICE.lower()
    sample = Path(settings.UPLOAD_FOLDER) / "warmup.wav"

    # Warm-up diarizer
    try:
        get_diarizer()(str(sample))
        logger.info("✅ Warm-up diarizer complete")
    except Exception as e:
        logger.warning(f"Warm-up diarizer failed: {e}")

    # Warm-up WhisperModel только на GPU
    if device != "cpu":
        try:
            whisper = get_whisper_model()
            whisper.transcribe(
                str(sample),
                offset=0,
                duration=2.0,
                language=settings.WHISPER_LANGUAGE,
                vad_filter=True
            )
            logger.info("✅ Warm-up WhisperModel complete")
        except Exception as e:
            logger.warning(f"Warm-up WhisperModel failed: {e}")


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for '{src}'")
    segments = split_audio_fixed_windows(src, settings.SEGMENT_LENGTH_S)
    logger.info(f" -> {len(segments)} segments of up to {settings.SEGMENT_LENGTH_S}s")

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f" Transcribing segment {idx}: {start:.1f}s → {end:.1f}s")
        result = whisper.transcribe(
            str(src),
            beam_size=settings.WHISPER_BATCH_SIZE,
            language=settings.WHISPER_LANGUAGE,
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
    out_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Transcription complete: saved to '{out_path}'")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
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
    out_path.write_text(
        json.dumps(speakers, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Diarization complete: saved to '{out_path}'")


def split_audio_fixed_windows(audio_path: Path, window_s: int):
    audio = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    segments = []
    for start_ms in range(0, length_ms, window_ms):
        end_ms = min(start_ms + window_ms, length_ms)
        segments.append((start_ms / 1000.0, end_ms / 1000.0))
    return segments