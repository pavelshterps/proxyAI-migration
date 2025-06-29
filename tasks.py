# tasks.py
import os, logging
from pathlib import Path

from celery import shared_task
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment, silence

from config.settings import UPLOAD_FOLDER, RESULTS_FOLDER

logger = logging.getLogger(__name__)

_whisper_model = None
_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = os.getenv("WHISPER_MODEL_PATH",
                               "/hf_cache/models--guillaumekln--faster-whisper-medium")
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        logger.info(f"Loading WhisperModel: {model_path} on {device} ({compute_type})")
        _whisper_model = WhisperModel(model_path,
                                     device=device,
                                     compute_type=compute_type)
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = os.getenv("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading diarizer into cache {cache_dir} …")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=cache_dir
        )
        logger.info("Diarizer loaded")
    return _diarizer

def split_audio_on_silence(path: str,
                           min_silence_len=700,
                           silence_thresh=-40,
                           keep_silence=200):
    """
    Разбить WAV на кластеры по тишине.
    Возвращает список (start_sec, end_sec).
    """
    audio = AudioSegment.from_file(path, format="wav")
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    boundaries = []
    pos = 0
    for chunk in chunks:
        start = pos / 1000.0
        duration = len(chunk) / 1000.0
        boundaries.append((start, start + duration))
        pos += duration * 1000
    return boundaries

@shared_task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str):
    whisper = get_whisper_model()
    audio_file = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {audio_file}")
    segments = split_audio_on_silence(str(audio_file))
    if not segments:
        segments = [(0.0, None)]

    transcript = []
    for idx, (start, end) in enumerate(segments):
        logger.debug(f"Segment {idx}: {start}-{end}")
        kwargs = dict(beam_size=5,
                      language="ru",
                      vad_filter=True,
                      word_timestamps=True)
        if start is not None:
            kwargs["offset"] = start
        if end is not None:
            kwargs["duration"] = end - start

        result = whisper.transcribe(str(audio_file), **kwargs)
        text = " ".join([seg.text for seg in result.segments])
        transcript.append({
            "segment": idx,
            "start": start,
            "end": end,
            "text": text
        })

    import json
    with open(out_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved transcript to {out_dir/'transcript.json'}")

@shared_task(name="tasks.diarize_full")
def diarize_full(upload_id: str):
    diarizer = get_diarizer()
    audio_file = Path(UPLOAD_FOLDER) / f"{upload_id}.wav"
    out_dir = Path(RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Diarizing {audio_file}")
    diarization = diarizer(str(audio_file))

    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    import json
    with open(out_dir / "diarization.json", "w", encoding="utf-8") as f:
        json.dump(speakers, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved diarization to {out_dir/'diarization.json'}")