# tasks.py
import os, json, logging
from pathlib import Path
from celery import shared_task
from faster_whisper import WhisperModel, load_vad
from pyannote.audio import Pipeline
from config.settings import settings

logger = logging.getLogger(__name__)
_whisper_model = None
_diarizer = None
_vad = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading WhisperModel at '{settings.WHISPER_MODEL_PATH}'")
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_PATH,
            device=settings.WHISPER_DEVICE,
            device_index=settings.WHISPER_DEVICE_INDEX,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        _vad = load_vad()
    return _vad

def split_audio(src: Path):
    # attempt real VAD
    try:
        vad = get_vad()
        segments = vad(str(src), chunk_length_s=settings.SEGMENT_LENGTH_S)
        if segments:
            return segments
    except Exception as e:
        logger.warning(f"VAD failed: {e}; falling back to fixed windows")

    # fallback fixed windows
    length = WhisperModel.get_audio_duration(str(src))
    seg = settings.SEGMENT_LENGTH_S
    return [(i, min(i+seg, length)) for i in range(0, int(length), seg)]

@shared_task(
    bind=True,
    name="tasks.transcribe_segments",
    autoretry_for=(IOError, RuntimeError),
    retry_backoff=True,
    max_retries=2,
)
def transcribe_segments(self, upload_id: str):
    whisper = get_whisper_model()
    src = Path(settings.UPLOAD_FOLDER)/f"{upload_id}.wav"
    out_dir = Path(settings.RESULTS_FOLDER)/upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{upload_id}] Transcribing {src}")
    segments = split_audio(src)
    logger.info(f"[{upload_id}] {len(segments)} segments")

    transcript = []
    for i,(start,end) in enumerate(segments):
        try:
            res = whisper.transcribe(
                str(src),
                task=settings.WHISPER_TASK,
                language="ru",
                beam_size=settings.WHISPER_BEAM_SIZE,
                offset=start, duration=(end-start),
                vad_filter=False,
            )
            text = res["segments"][0]["text"]
        except Exception as e:
            logger.error(f"Segment {i} failed: {e}")
            text = ""
        transcript.append({"segment":i,"start":start,"end":end,"text":text})

    with open(out_dir/"transcript.json","w",encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    logger.info(f"[{upload_id}] Transcript saved")

@shared_task(
    bind=True,
    name="tasks.diarize_full",
    autoretry_for=(IOError, RuntimeError),
    retry_backoff=True,
    max_retries=2,
)
def diarize_full(self, upload_id: str):
    global _diarizer
    if _diarizer is None:
        os.makedirs(settings.DIARIZER_CACHE_DIR, exist_ok=True)
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL, cache_dir=settings.DIARIZER_CACHE_DIR
        )

    src = Path(settings.UPLOAD_FOLDER)/f"{upload_id}.wav"
    out_dir = Path(settings.RESULTS_FOLDER)/upload_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{upload_id}] Diarizing {src}")
    diarization = _diarizer(str(src))
    speakers = [
        {"start":t.start,"end":t.end,"speaker":sp}
        for t,_,sp in diarization.itertracks(yield_label=True)
    ]

    with open(out_dir/"diarization.json","w",encoding="utf-8") as f:
        json.dump(speakers, f, ensure_ascii=False, indent=2)
    logger.info(f"[{upload_id}] Diarization saved")