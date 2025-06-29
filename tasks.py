import os
import json
import structlog
import webrtcvad
from pathlib import Path
from collections import deque

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from pydub import AudioSegment

from config.settings import settings

logger = structlog.get_logger(__name__)

# Singletons для ленивой загрузки моделей
_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading WhisperModel",
            path=settings.whisper_model_path,
            device=settings.whisper_device,
            compute=settings.whisper_compute_type,
        )
        _whisper_model = WhisperModel(
            settings.whisper_model_path,
            device=settings.whisper_device,
            device_index=settings.whisper_device_index,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.makedirs(settings.diarizer_cache_dir, exist_ok=True)
        logger.info(
            "Loading diarizer",
            protocol=settings.pyannote_protocol,
            cache_dir=str(settings.diarizer_cache_dir),
        )
        _diarizer = Pipeline.from_pretrained(
            settings.pyannote_protocol,
            cache_dir=str(settings.diarizer_cache_dir),
        )
        logger.info("Diarizer loaded")
    return _diarizer


def vad_segment_generator(
    audio_path: Path, aggressiveness: int = 3, frame_duration_ms: int = 30
):
    """
    Разбивает WAV-файл на участки речи по VAD.
    Возвращает генератор сегментов (start_s, end_s).
    """
    # читаем audio
    audio = AudioSegment.from_file(str(audio_path)).set_frame_rate(16000).set_channels(1)
    pcm_data = audio.raw_data
    sample_rate = audio.frame_rate

    vad = webrtcvad.Vad(aggressiveness)

    frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit = 2 bytes
    offset = 0
    frames = []
    timestamps = []

    # разбиваем на кадры
    while offset + frame_bytes <= len(pcm_data):
        frame = pcm_data[offset : offset + frame_bytes]
        timestamp_start = offset / (sample_rate * 2)
        timestamp_end = (offset + frame_bytes) / (sample_rate * 2)
        is_speech = vad.is_speech(frame, sample_rate)
        frames.append(is_speech)
        timestamps.append((timestamp_start, timestamp_end))
        offset += frame_bytes

    # собираем сегменты речи по оконной агрегации
    min_speech_frames = int(200 / frame_duration_ms)  # минимум 200 ms речи
    min_silence_frames = int(300 / frame_duration_ms)  # минимум 300 ms тишины

    speech_queue = deque(maxlen=min_silence_frames)
    triggered = False
    seg_start = 0.0

    for idx, is_speech in enumerate(frames):
        if not triggered:
            speech_queue.append(is_speech)
            num_voiced = sum(speech_queue)
            if num_voiced > 0.9 * speech_queue.maxlen:
                triggered = True
                seg_start = timestamps[idx - speech_queue.maxlen + 1][0]
                speech_queue.clear()
        else:
            speech_queue.append(is_speech)
            num_unvoiced = speech_queue.maxlen - sum(speech_queue)
            if num_unvoiced > 0.9 * speech_queue.maxlen:
                seg_end = timestamps[idx - speech_queue.maxlen + 1][1]
                yield seg_start, seg_end
                triggered = False
                speech_queue.clear()

    # если файл кончился во время речи
    if triggered:
        yield seg_start, timestamps[-1][1]


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    """
    1) Разбивает аудио на речевые сегменты по VAD
    2) Транскрибирует каждый сегмент
    3) Сохраняет RESULTS/<upload_id>/transcript.json
    """
    whisper = get_whisper_model()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    out_dir = Path(settings.results_folder) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = list(vad_segment_generator(src))
    logger.info("Transcribing", upload_id=upload_id, segments=len(segments))

    transcript = []
    for i, (start, end) in enumerate(segments):
        logger.debug(f"Segment {i}: {start:.2f}s–{end:.2f}s")
        result = whisper.transcribe(
            str(src),
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language,
            vad_filter=False,  # уже применили VAD
            word_timestamps=True,
            offset=start,
            duration=end - start,
        )
        text = result["segments"][0]["text"]
        transcript.append(
            {"segment": i, "start": start, "end": end, "text": text}
        )

    out_file = out_dir / "transcript.json"
    out_file.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), "utf-8"
    )
    logger.info("Transcription saved", path=str(out_file), upload_id=upload_id)


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    """
    1) Выполняет диаризацию всего файла
    2) Сохраняет RESULTS/<upload_id>/diarization.json
    """
    diarizer = get_diarizer()
    src = Path(settings.upload_folder) / f"{upload_id}.wav"
    out_dir = Path(settings.results_folder) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Diarization started", upload_id=upload_id)
    diar = diarizer(str(src))

    speakers = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        speakers.append(
            {"start": turn.start, "end": turn.end, "speaker": spk}
        )

    out_file = out_dir / "diarization.json"
    out_file.write_text(
        json.dumps(speakers, ensure_ascii=False, indent=2), "utf-8"
    )
    logger.info("Diarization saved", path=str(out_file), upload_id=upload_id)