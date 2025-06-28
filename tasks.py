import os
import logging
import json
from celery import Celery
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)
app = Celery("proxyai")
app.config_from_object("config")

# Cache directory for diarization models (writable inside container)
DIARIZER_CACHE = os.environ.get("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")
# Directory to write results (ensure volume mount exists)
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/tmp/results")

_whisper_model = None
_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = os.environ.get(
            "WHISPER_MODEL_PATH",
            "/hf_cache/models--guillaumekln--faster-whisper-medium"
        )
        logger.info(
            "Loading WhisperModel once at startup: "
            f"{{'model_size_or_path': model_path, 'device': 'cuda', "
            "'compute_type': 'int8', 'device_index': 0}}"
        )
        _whisper_model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="int8",
            device_index=0
        )
        logger.info("WhisperModel loaded (quantized int8)")
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.makedirs(DIARIZER_CACHE, exist_ok=True)
        logger.info(f"Loading Pyannote diarizer with cache_dir={DIARIZER_CACHE}")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=DIARIZER_CACHE,
            use_auth_token=True
        )
        logger.info("Pyannote diarizer loaded")
    return _diarizer

def _ensure_results_dir(upload_id):
    path = os.path.join(RESULTS_DIR, upload_id)
    os.makedirs(path, exist_ok=True)
    return path

@app.task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str, segments: list):
    """
    Transcribe provided audio segments with WhisperModel.
    """
    model = get_whisper_model()
    results = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        path = seg["path"]
        logger.info(f"Transcribing segment {path} [{start}-{end}]")
        segments_result, info = model.transcribe(
            path,
            beam_size=5,
            language="ru"
        )
        # Concatenate text pieces
        text = "".join([piece for piece, _ in segments_result])
        results.append({"start": start, "end": end, "text": text})
    out_dir = _ensure_results_dir(upload_id)
    out_file = os.path.join(out_dir, "transcription.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved transcription for {upload_id} to {out_file}")
    return results

@app.task(name="tasks.diarize_full")
def diarize_full(upload_id: str, audio_path: str):
    """
    Perform speaker diarization on full audio file.
    """
    diarizer = get_diarizer()
    logger.info(f"Running diarization for {audio_path}")
    diarization = diarizer(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    out_dir = _ensure_results_dir(upload_id)
    out_file = os.path.join(out_dir, "diarization.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved diarization for {upload_id} to {out_file}")
    return segments