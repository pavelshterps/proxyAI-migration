# tasks.py
import os
import logging

import whisperx
from celery_app import celery
from celery.result import AsyncResult
from config.settings import settings

from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

logger = logging.getLogger(__name__)

# Caches
_whisper_model = None
_whisper_processor = None
_align_model = None
_align_metadata = None

def get_whisper_model():
    """
    Load (or get cached) 4-bit quantized Whisper model + processor.
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
    Load (or get cached) aligner model + metadata.
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
    Transcribe & align a single audio chunk, shifting timestamps by offset.
    """
    model, processor = get_whisper_model()
    align_model, align_metadata = get_align_model()

    # Load audio
    audio = AudioSegment.from_file(chunk_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_array = samples / (1 << (8 * audio.sample_width - 1))

    # ASR
    inputs = processor(audio_array, return_tensors="pt", sampling_rate=sr).to(model.device)
    tokens = model.generate(**inputs)
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0]

    # Prepare single segment for alignment
    segment = [{"start": 0.0, "end": len(audio_array)/sr, "text": text}]

    # Alignment
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
            "end":   seg.end + offset,
            "speaker": getattr(seg, "speaker", seg.speaker_label),
            "text": seg.text
        })

    # Cleanup chunk file
    try:
        os.remove(chunk_path)
    except OSError:
        pass

    return out_segs

@celery.task(bind=True)
def transcribe_task(self, audio_path: str):
    """
    1. Split the long audio on silence (and into 3min max subchunks)
    2. Dispatch transcribe_chunk for each
    3. Merge & sort the results
    """
    # Split on silence
    sound = AudioSegment.from_file(audio_path)
    raw_chunks = split_on_silence(
        sound,
        min_silence_len=1000,
        silence_thresh=sound.dBFS - 16,
        keep_silence=500
    )

    # Export chunks and schedule subtasks
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

    # Run all subtasks in parallel
    group_result = celery.group(tasks).apply_async()
    chunk_results = group_result.get()  # list of lists

    # Merge and sort
    merged = [seg for result in chunk_results for seg in result]
    merged.sort(key=lambda x: x["start"])

    return {
        "segments": merged,
        "audio_filepath": audio_path
    }

@celery.task
def cleanup_files(path: str):
    """Remove temporary files."""
    try:
        os.remove(path)
    except OSError:
        pass

def get_file_path_by_task_id(task_id: str) -> str:
    """
    Retrieve the audio_filepath from a completed task.
    """
    res = AsyncResult(task_id, app=celery)
    if res.state == 'SUCCESS':
        payload = res.get()
        return payload.get('audio_filepath')
    return None