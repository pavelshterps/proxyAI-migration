import os
import json
import wave
import collections
from pathlib import Path

import structlog
import webrtcvad
from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import settings, Settings

# Structlog
logger = structlog.get_logger()

# Синглтоны моделей в рамках процесса
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading WhisperModel",
            path=settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.makedirs(settings.DIARIZER_CACHE_DIR, exist_ok=True)
        logger.info("Loading pyannote Pipeline", cache_dir=settings.DIARIZER_CACHE_DIR)
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=settings.DIARIZER_CACHE_DIR,
        )
        logger.info("Diarizer loaded")
    return _diarizer


def read_wave(path: str):
    """Возвращает (pcm_data, sample_rate). WAV должен быть mono 16-bit."""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        assert wf.getcomptype() == "NONE"
        pcm = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    return pcm, rate


def frame_generator(frame_ms: int, audio: bytes, rate: int):
    n = int(rate * frame_ms / 1000) * 2
    offset = 0
    timestamp = 0.0
    duration = (n / 2) / rate
    while offset + n <= len(audio):
        yield audio[offset : offset + n], timestamp
        timestamp += duration
        offset += n


def vad_collector(
    rate: int,
    frame_ms: int,
    padding_ms: int,
    vad: webrtcvad.Vad,
    frames,
):
    num_padding = padding_ms // frame_ms
    ring = collections.deque(maxlen=num_padding)
    triggered = False
    segments = []
    voiced_start = 0.0

    for frame, ts in frames:
        is_speech = vad.is_speech(frame, rate)
        if not triggered:
            ring.append((frame, ts, is_speech))
            num_voiced = sum(1 for (_, _, speech) in ring if speech)
            if num_voiced > 0.9 * ring.maxlen:
                triggered = True
                voiced_start = ring[0][1]
                ring.clear()
        else:
            if not is_speech:
                ring.append((frame, ts, is_speech))
                num_unvoiced = sum(1 for (_, _, speech) in ring if not speech)
                if num_unvoiced > 0.9 * ring.maxlen:
                    end = ts + (frame_ms / 1000.0)
                    segments.append((voiced_start, end))
                    triggered = False
                    ring.clear()
            else:
                # голос продолжается
                ring.clear()
    # хвост
    if triggered:
        segments.append((voiced_start, ts + frame_ms / 1000.0))
    return segments


def split_audio_vad(audio_path: Path):
    pcm, rate = read_wave(str(audio_path))
    vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
    frames = list(frame_generator(settings.VAD_FRAME_MS, pcm, rate))
    segments = vad_collector(
        rate,
        settings.VAD_FRAME_MS,
        settings.VAD_PADDING_MS,
        vad,
        frames,
    )
    logger.info("VAD segmentation produced segments", count=len(segments))
    return segments if segments else [(0.0, None)]


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst = Path(settings.RESULTS_FOLDER) / upload_id
    dst.mkdir(parents=True, exist_ok=True)

    logger.info("Starting diarization", upload_id=upload_id, path=str(src))
    diarization = diarizer(str(src))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    out = dst / "diarization.json"
    out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Diarization complete", path=str(out))


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    dst = Path(settings.RESULTS_FOLDER) / upload_id
    dst.mkdir(parents=True, exist_ok=True)

    logger.info("Starting transcription", upload_id=upload_id, path=str(src))
    # сегментация по VAD
    segments = split_audio_vad(src)

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug("Transcribing segment", idx=idx, start=start, end=end)
        result = whisper.transcribe(
            str(src),
            beam_size=5,
            language="ru",
            vad_filter=False,  # уже сделали VAD сами
            word_timestamps=True,
            offset=start,
            duration=None if end is None else (end - start),
        )
        text = result["segments"][0]["text"]
        transcript.append({"segment": idx, "start": start, "end": end, "text": text})

    out = dst / "transcript.json"
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Transcription complete", path=str(out))