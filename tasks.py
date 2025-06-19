import os
import time
import logging

import whisperx
import torch
import numpy as np
from celery import chord
from celery_app import celery
from celery.result import AsyncResult
from config.settings import settings
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Кэш моделей и процессора
_whisper_model = None
_whisper_processor = None
_align_model = None
_align_metadata = None

def estimate_processing_time(audio_path: str, speed_factor: float = 1.0) -> float:
    """
    Оценка времени обработки (сек) по длине аудио.
    speed_factor=1.0 — реальное время, <1 — быстрее, >1 — медленнее.
    len(audio) даёт миллисекунды, делим на 1000 → секунды.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # секунды
        return duration * speed_factor
    except Exception:
        return 0.0

def get_whisper_model():
    """
    Load (and cache) the 4-bit quantized Whisper model and processor.
    Map settings.DEVICE ('gpu'→'cuda'), catch OOM or internal errors and fallback to CPU int8.
    """
    global _whisper_model, _whisper_processor
    if _whisper_model is None:
        # нормализуем DEVICE
        dev = settings.DEVICE.lower()
        if dev == 'gpu':
            dev = 'cuda'
        if dev not in ('cpu', 'cuda'):
            dev = 'cpu'

        # собираем конфиг для quantization без max_memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
        )
        try:
            _whisper_model = WhisperForConditionalGeneration.from_pretrained(
                settings.WHISPER_MODEL,
                token=settings.HUGGINGFACE_TOKEN,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError, AttributeError) as e:
            logger.warning(f"Quantized GPU load failed ({e}); falling back to CPU int8.")
            _whisper_model = WhisperForConditionalGeneration.from_pretrained(
                settings.WHISPER_MODEL,
                token=settings.HUGGINGFACE_TOKEN,
                load_in_8bit=True,
                device_map="cpu"
            )

    if _whisper_processor is None:
        _whisper_processor = WhisperProcessor.from_pretrained(
            settings.WHISPER_MODEL,
            token=settings.HUGGINGFACE_TOKEN
        )

    return _whisper_model, _whisper_processor

def get_align_model():
    """
    Load (and cache) the WhisperX align model and its metadata.
    Converts settings.DEVICE='gpu'→'cuda', falls back to CPU on error.
    """
    global _align_model, _align_metadata
    if _align_model is None or _align_metadata is None:
        dev = settings.DEVICE.lower()
        if dev == 'gpu':
            dev = 'cuda'
        if dev not in ('cpu', 'cuda'):
            dev = 'cpu'

        try:
            _align_model, _align_metadata = whisperx.load_align_model(
                language_code=settings.LANGUAGE_CODE,
                device=dev
            )
        except RuntimeError as e:
            logger.warning(f"Failed to load align model on '{dev}': {e}; falling back to CPU.")
            _align_model, _align_metadata = whisperx.load_align_model(
                language_code=settings.LANGUAGE_CODE,
                device='cpu'
            )

    return _align_model, _align_metadata

@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def transcribe_chunk(self, chunk_path: str, offset: float):
    """
    Transcribe and align a single audio chunk, shifting timestamps by offset.
    """
    start_ts = time.time()
    logger.info(f"Starting chunk transcription: {chunk_path} (offset {offset}s)")

    model, processor = get_whisper_model()
    align_model, align_metadata = get_align_model()

    audio = AudioSegment.from_file(chunk_path)
    audio = audio.set_frame_rate(16000)
    sr = 16000

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_array = samples / (1 << (8 * audio.sample_width - 1))

    inputs = processor(audio_array, return_tensors="pt", sampling_rate=sr).to(model.device)
    tokens = model.generate(**inputs)
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0]

    segments = [{"start": 0.0, "end": len(audio_array) / sr, "text": text}]

    aligned = whisperx.align(
        segments,
        processor.tokenizer,
        chunk_path,
        align_model,
        align_metadata,
        device=model.device
    )

    out = []
    for seg in aligned["segments"]:
        out.append({
            "start": seg.start + offset,
            "end":   seg.end   + offset,
            "speaker": getattr(seg, "speaker", seg.speaker_label),
            "text": seg.text
        })

    try:
        os.remove(chunk_path)
    except OSError:
        pass

    elapsed = time.time() - start_ts
    logger.info(f"Finished chunk transcription: {chunk_path} in {elapsed:.1f}s")
    return out

@celery.task
def merge_chunks(results_list, audio_path: str):
    merged = [seg for sub in results_list for seg in sub]
    merged.sort(key=lambda x: x["start"])
    return {"segments": merged, "audio_filepath": audio_path}

@celery.task(bind=True)
def transcribe_task(self, audio_path: str):
    start_ts = time.time()
    estimate = estimate_processing_time(audio_path)
    logger.info(f"Starting overall transcription: {audio_path}, estimated ~{estimate:.1f}s")

    sound = AudioSegment.from_file(audio_path)
    raw_chunks = split_on_silence(
        sound,
        min_silence_len=1000,
        silence_thresh=sound.dBFS - 16,
        keep_silence=500
    )

    tasks = []
    elapsed_ms = 0
    chunk_ms = 3 * 60 * 1000
    for chunk in raw_chunks:
        start_ms = 0
        while start_ms < len(chunk):
            sub = chunk[start_ms:start_ms + chunk_ms]
            cp = f"{audio_path}.chunk_{elapsed_ms + start_ms}.wav"
            sub.export(cp, format="wav")
            tasks.append(transcribe_chunk.s(cp, (elapsed_ms + start_ms) / 1000.0))
            start_ms += chunk_ms
        elapsed_ms += len(chunk)

    task_result = chord(tasks)(merge_chunks.s(audio_path))
    logger.info(f"Dispatched chord with id {task_result.id}")
    return {"merge_task_id": task_result.id}

@celery.task
def cleanup_files(path: str):
    try:
        os.remove(path)
    except OSError:
        pass

def get_file_path_by_task_id(task_id: str) -> str:
    res = AsyncResult(task_id, app=celery)
    if res.state == 'SUCCESS':
        return res.get().get('audio_filepath')
    return None