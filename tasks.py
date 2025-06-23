from typing import List, Dict
import os
import glob
import logging

from celery import chord, group
from celery_app import celery_app
from config.settings import (
    HUGGINGFACE_TOKEN,
    HF_CACHE_DIR,
    PYANNOTE_PROTOCOL,
    ALIGN_BEAM_SIZE,
)
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Lazy singleton loaders
_diarizer = None
_whisper = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        _diarizer = Pipeline.from_pretrained(
            PYANNOTE_PROTOCOL,
            use_auth_token=HUGGINGFACE_TOKEN,
            cache_dir=HF_CACHE_DIR
        )
    return _diarizer

def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(
            os.getenv("WHISPER_MODEL"),
            device=os.getenv("DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float32"),
            cache_dir=HF_CACHE_DIR
        )
    return _whisper

@celery_app.task(name="tasks.diarize_task", queue="preprocess")
def diarize_task(filepath: str) -> List[Dict]:
    logger.info("Diarization start: %s", filepath)
    diarizer = get_diarizer()
    segments = diarizer(filepath)
    # return list of {start,end}
    result = [{"start": s.start, "end": s.end} for s, _, _ in segments.itertracks(yield_label=True)]
    logger.info("Diarization done: %d segments", len(result))
    return result

@celery_app.task(name="tasks.chunk_task", queue="preprocess")
def chunk_task(filepath_segments) -> List[str]:
    filepath, segments = filepath_segments
    logger.info("Chunking %s into %d parts", filepath, len(segments))
    audio = AudioSegment.from_file(filepath)
    out_paths = []
    os.makedirs("/tmp/chunks", exist_ok=True)
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        chunk = audio[start_ms:end_ms]
        out_file = f"/tmp/chunks/{start_ms}_{end_ms}.wav"
        chunk.export(out_file, format="wav")
        out_paths.append(out_file)
    logger.info("Chunking done: %d files", len(out_paths))
    return out_paths

@celery_app.task(name="tasks.inference_task", queue="inference")
def inference_task(chunk_path: str) -> Dict:
    logger.info("Inference on %s", chunk_path)
    model = get_whisper()
    segments, _ = model.transcribe(
        chunk_path,
        beam_size=ALIGN_BEAM_SIZE,
        word_timestamps=False,
    )
    text = "".join(seg.text for seg in segments)
    return {"chunk_path": chunk_path, "text": text}

@celery_app.task(name="tasks.merge_results", queue="preprocess")
def merge_results(results: List[Dict], original: str) -> Dict:
    logger.info("Merging %d results for %s", len(results), original)
    # collect & sort by chunk_path
    texts = [r["text"] for r in sorted(results, key=lambda r: r["chunk_path"])]
    full = "\n".join(texts)
    # cleanup
    for f in glob.glob("/tmp/chunks/*.wav"):
        try: os.remove(f)
        except: pass
    try: os.remove(original)
    except: pass
    return {"text": full}

@celery_app.task(name="tasks.transcribe_full", queue="preprocess")
def transcribe_full(filepath: str) -> str:
    """
    1) diarize → 2) chunk → 3) parallel inference → 4) merge → returns merge_task_id
    """
    logger.info("Orchestrating full transcription: %s", filepath)

    # chain preprocess: get segments
    diag = diarize_task.s(filepath)

    # then chunk: we need both filepath and segments
    chnk = chunk_task.s(filepath)

    # inference group on each chunk
    inf_group = group(inference_task.s() for _ in range(0))  # placeholder

    # but Celery chord needs to know number of elements;
    # instead we use chord in callback of chunk:
    def build_chord(chunks):
        return chord(
            group(inference_task.s(p) for p in chunks),
            merge_results.s(filepath)
        )()
    # use .then on chunk task result
    ev = chnk.apply_async(link=build_chord)

    logger.info("Dispatched chord callback, id=%s", ev.id)
    return ev.id