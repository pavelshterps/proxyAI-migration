import os
import time
import logging

import whisperx
import torch
import numpy as np
from celery import group
from celery_app import celery
from celery.result import AsyncResult
from config.settings import settings
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Cache models & processor to avoid reloading for each task
_whisper_model = None
_whisper_processor = None
_align_model = None
_align_metadata = None

def estimate_processing_time(audio_path: str, speed_factor: float = 1.0) -> float:
    """
    Estimate wall-clock time in seconds for processing given audio.
    speed_factor: 1.0 means real-time, <1 faster, >1 slower.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # in seconds
        return duration * speed_factor
    except Exception:
        return 0.0

def get_whisper_model():
    """
    Load (or return cached) 4-bit quantized Whisper model + processor.
    Uses bitsandbytes for 4-bit and offloading to fit in <4GB VRAM.
    """
    global _whisper_model, _whisper_processor
    if _whisper_model is None:
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(
            settings.WHISPER_MODEL,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            device_map="auto",
            torch_dtype=torch.float16,
            offload_state_dict=True,
            offload_folder="offload"
        )
        _whisper_processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
    return _whisper_model, _whisper_processor

def get_align_model():
    """
    Load (or return cached) WhisperX align model + metadata.
    """
    global _align_model, _align_metadata
    if _align_model is None or _align_metadata is None:
        device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code=settings.LANGUAGE_CODE,
            device=device
        )
    return _align_model, _align_metadata

@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def transcribe_chunk(self, chunk_path: str, offset: float):
    """
    Transcribe & align a single audio chunk, shifting timestamps by offset (seconds).
    """
    start_ts = time.time()
    logger.info(f"Starting chunk transcription: {chunk_path} (offset {offset}s)")
    model, processor = get_whisper_model()
    align_model, align_metadata = get_align_model()

    # Load and normalize audio
    audio = AudioSegment.from_file(chunk_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_array = samples / (1 << (8 * audio.sample_width - 1))

    # ASR: generate tokens → decode to text
    inputs = processor(audio_array, return_tensors="pt", sampling_rate=sr).to(model.device)
    tokens = model.generate(**inputs)
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0]

    # Prepare single segment for alignment
    segment = [{"start": 0.0, "end": len(audio_array) / sr, "text": text}]

    # Alignment: whisperx
    device = settings.DEVICE.lower() if isinstance(settings.DEVICE, str) else settings.DEVICE
    aligned = whisperx.align(
        segment,
        processor.tokenizer,
        chunk_path,
        align_model,
        align_metadata,
        device=device
    )

    # Shift times by offset
    out_segs = []
    for seg in aligned["segments"]:
        out_segs.append({
            "start": seg.start + offset,
            "end":   seg.end   + offset,
            "speaker": getattr(seg, "speaker", seg.speaker_label),
            "text": seg.text
        })

    # Cleanup temp chunk file
    try:
        os.remove(chunk_path)
    except OSError:
        pass

    elapsed = time.time() - start_ts
    logger.info(f"Finished chunk transcription: {chunk_path} in {elapsed:.1f}s")
    return out_segs

@celery.task(bind=True)
def transcribe_task(self, audio_path: str):
    """
    Orchestrator:
      1. Split audio on silence & into max-3-min subchunks
      2. Dispatch transcribe_chunk for each
      3. Merge & sort all segments
    """
    start_ts = time.time()
    estimate = estimate_processing_time(audio_path, speed_factor=1.0)
    logger.info(f"Starting overall transcription: {audio_path}, estimated ~{estimate:.1f}s")

    # 1) Split on silence
    sound = AudioSegment.from_file(audio_path)
    raw_chunks = split_on_silence(
        sound,
        min_silence_len=1000,
        silence_thresh=sound.dBFS - 16,
        keep_silence=500
    )

    # 2) Export chunks & schedule subtasks
    tasks = []
    elapsed_ms = 0
    max_len_ms = 3 * 60 * 1000  # 3 minutes

    for chunk in raw_chunks:
        start_ms = 0
        while start_ms < len(chunk):
            sub = chunk[start_ms : start_ms + max_len_ms]
            cp = f"{audio_path}.chunk_{elapsed_ms + start_ms}.wav"
            sub.export(cp, format="wav")
            tasks.append(transcribe_chunk.s(cp, (elapsed_ms + start_ms) / 1000.0))
            start_ms += max_len_ms
        elapsed_ms += len(chunk)

    # 3) Run all in parallel & wait
    job = group(tasks)
    group_result = job.apply_async()
    chunk_results = group_result.get()  # List[List[segment]]

    # 4) Merge & sort final segments
    merged = [seg for sublist in chunk_results for seg in sublist]
    merged.sort(key=lambda x: x["start"])

    total = time.time() - start_ts
    logger.info(f"Completed overall transcription: {audio_path} in {total:.1f}s")
    return {
        "segments": merged,
        "audio_filepath": audio_path
    }

@celery.task
def cleanup_files(path: str):
    """Remove temporary files after processing."""
    try:
        os.remove(path)
    except OSError:
        pass

def get_file_path_by_task_id(task_id: str) -> str:
    """
    Retrieve the audio_filepath from a completed transcription task.
    Returns None if task not SUCCESS.
    """
    res = AsyncResult(task_id, app=celery)
    if res.state == 'SUCCESS':
        payload = res.get()
        return payload.get('audio_filepath')
    return None