import os
from celery_app import app
from config.settings import settings
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# ленивые синглтоны
_diarizer = None
_whisper = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.makedirs(settings.HF_CACHE_DIR, exist_ok=True)
        os.environ["HF_HOME"] = settings.HF_CACHE_DIR
        _diarizer = Pipeline.from_pretrained(settings.PYANNOTE_MODEL)
    return _diarizer

def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(
            model=settings.WHISPER_MODEL,
            device=settings.DEVICE_TYPE,
            compute_type=settings.COMPUTE_TYPE,
            device_index=0,
            inter_threads=1,
            intra_threads=1,
        )
    return _whisper

@app.task(name="tasks.diarize_full")
def diarize_full(filepath: str):
    """
    делим полный файл на speaker-сегменты и шлём каждый сегмент на транскрипцию
    """
    diarizer = get_diarizer()
    diarization = diarizer(filepath)
    segments = []
    # собираем спикер-сегменты
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    # вырезаем каждый сегмент через ffmpeg и отдаём в очередь GPU
    for seg in segments:
        s, e, sp = seg["start"], seg["end"], seg["speaker"]
        out_path = f"{filepath}_{int(s*1000)}_{int(e*1000)}.wav"
        os.system(f"ffmpeg -y -i {filepath} -ss {s} -to {e} {out_path}")
        app.send_task("tasks.transcribe_segments", args=[out_path, sp])
    return segments

@app.task(name="tasks.transcribe_segments")
def transcribe_segments(filepath: str, speaker: str):
    """
    транскрибация маленького чанка GPU-моделью
    """
    whisper = get_whisper()
    segments, info = whisper.transcribe(filepath)
    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "speaker": speaker,
            "text": seg.text
        })
    # чистим временный файл
    os.remove(filepath)
    return result