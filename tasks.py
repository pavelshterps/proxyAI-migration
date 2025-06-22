import os
import glob
import logging
import torch
from celery_app import celery_app
from config.settings import (
    LOAD_IN_8BIT,
    WHISPER_MODEL_NAME,
    ALIGN_BEAM_SIZE,
    HUGGINGFACE_TOKEN,
    HF_CACHE_DIR,
    PYANNOTE_PROTOCOL,
)
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Lazy singletons
_diarizer = None
_whisper = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = DiarizationPipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN,
            cache_dir=HF_CACHE_DIR
        )
    return _diarizer

def get_whisper_model():
    global _whisper
    if _whisper is None:
        # faster-whisper will place model files under cache_dir/<model_name>
        _whisper = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="int8" if LOAD_IN_8BIT else "float32",
            cache_dir=HF_CACHE_DIR
        )
    return _whisper

@celery_app.task(
    name="tasks.diarize_task",
    queue="preprocess",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def diarize_task(filepath: str):
    logger.info("Task %s: starting diarization", diarize_task.request.id)
    diarizer = get_diarizer()
    segments = diarizer(filepath)
    result = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in segments.itertracks(yield_label=True)
    ]
    logger.info(
        "Task %s: diarization done (%d segments)",
        diarize_task.request.id,
        len(result),
    )
    return result

@celery_app.task(
    name="tasks.chunk_by_diarization",
    queue="preprocess",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def chunk_by_diarization(filepath: str, segments):
    logger.info(
        "Task %s: chunking %d segments", chunk_by_diarization.request.id, len(segments)
    )
    audio = AudioSegment.from_file(filepath)
    out_paths = []
    os.makedirs("/tmp/chunks", exist_ok=True)
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        chunk = audio[start_ms:end_ms]
        out_file = f"/tmp/chunks/{start_ms}_{end_ms}.wav"
        chunk.export(out_file, format="wav")
        out_paths.append(out_file)
    logger.info(
        "Task %s: chunking done (%d files)",
        chunk_by_diarization.request.id,
        len(out_paths),
    )
    return out_paths

@celery_app.task(
    name="tasks.inference_task",
    queue="inference",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
)
def inference_task(chunk_path: str):
    logger.info("Task %s: inference on %s", inference_task.request.id, chunk_path)
    model = get_whisper_model()
    segments, _ = model.transcribe(
        chunk_path,
        beam_size=ALIGN_BEAM_SIZE,
        word_timestamps=False,
    )
    text = "".join(seg.text for seg in segments)
    logger.info("Task %s: inference done", inference_task.request.id)
    return {"chunk": chunk_path, "text": text}

@celery_app.task(
    name="tasks.merge_results",
    queue="preprocess",
)
def merge_results(results, original_filepath: str):
    logger.info(
        "Task %s: merging %d results",
        merge_results.request.id,
        len(results),
    )
    full_text = "\n".join(r["text"] for r in results)
    # cleanup chunks
    for f in glob.glob("/tmp/chunks/*.wav"):
        try:
            os.remove(f)
        except:
            pass
    try:
        os.remove(original_filepath)
    except:
        pass
    logger.info("Task %s: cleanup done", merge_results.request.id)
    return {"text": full_text}

@celery_app.task(
    name="tasks.transcribe_full",
    queue="preprocess",
)
def transcribe_full(filepath: str):
    logger.info("Task %s: orchestration start", transcribe_full.request.id)
    # sequential chain
    diag = diarize_task.s(filepath)
    chunk = chunk_by_diarization.s(filepath)
    infer = inference_task.si()
    merge = merge_results.s(filepath)
    # build chord: [infer(chunk1), infer(chunk2), ...] -> merge
    chord_header = (
        celery_app.group(
            chunk | infer
        )
        .chord(merge)
    )
    res = chord_header(filepath)
    logger.info(
        "Task %s: dispatched chord callback %s",
        transcribe_full.request.id,
        res.id,
    )
    return {"callback_task_id": res.id}