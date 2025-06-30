import os
import json
import logging
from pathlib import Path

import structlog
import webrtcvad
from prometheus_client import Counter, Histogram

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

# ——— structured logger ——————————————————————————————————————
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()

# ——— Prometheus metrics ————————————————————————————————————
TASKS_TOTAL = Counter(
    "proxyai_celery_tasks_total",
    "Total number of Celery tasks executed",
    ["task_name"]
)
TASKS_FAIL = Counter(
    "proxyai_celery_tasks_failed",
    "Total number of failed Celery tasks",
    ["task_name"]
)
TRANSCRIBE_DURATION = Histogram(
    "proxyai_transcribe_duration_seconds",
    "Time spent transcribing audio segments",
    ["task_name"]
)
DIARIZE_DURATION = Histogram(
    "proxyai_diarize_duration_seconds",
    "Time spent in speaker diarization",
    ["task_name"]
)
MODEL_LOAD_ERRORS = Counter(
    "proxyai_model_load_errors_total",
    "Number of model load failures",
    ["model_name"]
)

# ——— singletons ————————————————————————————————————————————
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            model_path = os.getenv(
                "WHISPER_MODEL_PATH",
                "/hf_cache/models--guillaumekln--faster-whisper-medium"
            )
            device = os.getenv("WHISPER_DEVICE", "cuda")
            compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            log.info(
                "Loading WhisperModel",
                path=model_path, device=device, compute=compute
            )
            _whisper_model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute
            )
            log.info("WhisperModel loaded")
        except Exception as e:
            MODEL_LOAD_ERRORS.labels("whisper").inc()
            log.error("Failed to load WhisperModel", error=str(e))
            raise
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        try:
            cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
            os.makedirs(cache_dir, exist_ok=True)
            log.info("Loading pyannote Pipeline", cache_dir=cache_dir)
            _diarizer = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                cache_dir=cache_dir
            )
            log.info("Diarizer loaded")
        except Exception as e:
            MODEL_LOAD_ERRORS.labels("diarizer").inc()
            log.error("Failed to load diarizer", error=str(e))
            raise
    return _diarizer


def vad_split(audio_path: Path, sample_rate=16000, frame_duration_ms=30, padding_duration_ms=300):
    """
    Use webrtcvad to split audio into speech regions.
    Returns list of (start_sec, end_sec) tuples.
    """
    audio = AudioSegment.from_file(str(audio_path)).set_frame_rate(sample_rate).set_channels(1)
    raw = audio.raw_data
    vad = webrtcvad.Vad(int(os.getenv("VAD_AGGRESSIVENESS", "2")))
    frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2  # 16‐bit samples = 2 bytes

    # generator of (timestamp, is_speech)
    def frame_generator():
        for i in range(0, len(raw), frame_bytes):
            chunk = raw[i : i + frame_bytes]
            timestamp = (i / 2) / sample_rate
            yield timestamp, vad.is_speech(chunk, sample_rate)

    segments = []
    speech_on = False
    start_time = 0.0
    padding_frames = int(padding_duration_ms / frame_duration_ms)

    # simple state machine
    silence_count = 0
    for timestamp, is_speech in frame_generator():
        if is_speech:
            if not speech_on:
                speech_on = True
                start_time = timestamp
            silence_count = 0
        else:
            if speech_on:
                silence_count += 1
                if silence_count > padding_frames:
                    end_time = timestamp
                    segments.append((start_time, end_time))
                    speech_on = False
    # catch trailing speech
    if speech_on:
        segments.append((start_time, audio.duration_seconds))
    return segments


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    TASKS_TOTAL.labels("transcribe_segments").inc()
    whisper = get_whisper_model()

    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    log.info("Transcription started", upload_id=upload_id)
    segments = vad_split(src)
    log.info("VAD segmentation produced segments", upload_id=upload_id, count=len(segments))

    transcript = []
    with TRANSCRIBE_DURATION.labels("transcribe_segments").time():
        try:
            for idx, (start, end) in enumerate(segments):
                log.debug("Transcribing segment", upload_id=upload_id, idx=idx, start=start, end=end)
                result = whisper.transcribe(
                    str(src),
                    beam_size=5,
                    language="ru",
                    vad_filter=True,
                    word_timestamps=True,
                    offset=start,
                    duration=end - start,
                )
                text = result["segments"][0]["text"]
                transcript.append({
                    "segment": idx, "start": start, "end": end, "text": text
                })
        except Exception as e:
            TASKS_FAIL.labels("transcribe_segments").inc()
            log.error("Transcription failed", upload_id=upload_id, error=str(e))
            raise

    out_path = dst_dir / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Transcription complete", upload_id=upload_id, path=str(out_path))


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    TASKS_TOTAL.labels("diarize_full").inc()
    diarizer = get_diarizer()

    src = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst_dir = Path(RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    log.info("Diarization started", upload_id=upload_id)
    with DIARIZE_DURATION.labels("diarize_full").time():
        try:
            diarization = diarizer(str(src))
        except Exception as e:
            TASKS_FAIL.labels("diarize_full").inc()
            log.error("Diarization failed", upload_id=upload_id, error=str(e))
            raise

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start, "end": turn.end, "speaker": speaker
        })

    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Diarization complete", upload_id=upload_id, path=str(out_path))