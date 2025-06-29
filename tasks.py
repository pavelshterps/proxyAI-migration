import os
import json
import logging
import wave

import webrtcvad
from pathlib import Path
from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = os.getenv("WHISPER_MODEL_PATH")
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        device_index = int(os.getenv("WHISPER_DEVICE_INDEX", "0"))
        logger.info(
            f"Loading WhisperModel once: path={model_path}, "
            f"device={device}, compute_type={compute_type}, device_index={device_index}"
        )
        _whisper_model = WhisperModel(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type
        )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading diarizer with cache_dir={cache_dir}")
        _diarizer = Pipeline.from_pretrained(
            os.getenv("PYANNOTE_PROTOCOL", "pyannote/speaker-diarization"),
            cache_dir=cache_dir
        )
        logger.info("Diarizer loaded")
    return _diarizer


def vad_segment_generator(audio_path: str,
                          sample_rate: int = 16000,
                          frame_duration_ms: int = 30,
                          padding_duration_ms: int = 300,
                          aggressiveness: int = 3):
    """
    Разбивает WAV-файл на сегменты речи.
    Возвращает список кортежей (start_sec, end_sec).
    """
    vad = webrtcvad.Vad(aggressiveness)

    # Открываем WAV и проверяем параметры
    wf = wave.open(audio_path, "rb")
    assert wf.getnchannels() == 1, "Audio must be mono"
    assert wf.getsampwidth() == 2, "Audio must be 16-bit"
    assert wf.getframerate() == sample_rate, f"Sample rate must be {sample_rate}"

    frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
    frame_bytes = frame_size * 2  # bytes per frame (16-bit)

    raw_data = wf.readframes(frame_size)
    timestamp = 0.0
    speech_segments = []
    current_start = None
    padding_frames = int(padding_duration_ms / frame_duration_ms)

    silence_counter = 0

    while len(raw_data) == frame_bytes:
        is_speech = vad.is_speech(raw_data, sample_rate)
        if is_speech:
            if current_start is None:
                current_start = timestamp
            silence_counter = 0
        else:
            if current_start is not None:
                silence_counter += 1
                if silence_counter >= padding_frames:
                    end = timestamp + frame_duration_ms / 1000.0
                    speech_segments.append((current_start, end))
                    current_start = None
                    silence_counter = 0
        timestamp += frame_duration_ms / 1000.0
        raw_data = wf.readframes(frame_size)

    # Если до конца файла всё ещё был режим речи
    if current_start is not None:
        speech_segments.append((current_start, timestamp))

    wf.close()
    return speech_segments or [(0.0, None)]


@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    """
    Разбивает аудио по VAD, транскрибирует каждый сегмент Faster-Whisper,
    сохраняет transcript.json в RESULTS_FOLDER/<upload_id>.
    """
    whisper = get_whisper_model()

    audio_file = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    output_dir = Path(RESULTS_FOLDER) / upload_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting transcription for upload {upload_id} ({audio_file})")

    # сегментация по речи через webrtcvad
    segments = vad_segment_generator(str(audio_file))
    logger.info(f"VAD produced {len(segments)} segments")

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"Transcribing segment {idx}: {start:.2f}-{end or 'EOF'}")
        result = whisper.transcribe(
            str(audio_file),
            beam_size=int(os.getenv("WHISPER_BEAM_SIZE", 5)),
            language="ru",
            offset=start,
            duration=None if end is None else (end - start),
            vad_filter=False,            # уже отфильтровано
            word_timestamps=True
        )
        text = result["segments"][0]["text"]
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text
        })

    with open(output_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    logger.info(f"Transcription complete, saved to {output_dir/'transcript.json'}")


@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    """
    Полная диаризация файла pyannote.audio, сохраняем diarization.json.
    """
    diarizer = get_diarizer()

    audio_file = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    output_dir = Path(RESULTS_FOLDER) / upload_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diarization for upload {upload_id} ({audio_file})")
    diarization = diarizer(str(audio_file))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    with open(output_dir / "diarization.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Diarization complete, saved to {output_dir/'diarization.json'}")