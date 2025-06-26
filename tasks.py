import os
import subprocess
from celery_app import celery_app
from config.settings import settings

# ленивые загрузки
_diarizer = None
_whisper = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        from pyannote.audio import Pipeline
        os.environ["HF_HOME"] = settings.HF_CACHE
        _diarizer = Pipeline.from_pretrained(settings.PYANNOTE_MODEL)
    return _diarizer

def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        os.environ["HF_HOME"] = settings.HF_CACHE
        _whisper = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE_INDEX,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            inter_threads=settings.WHISPER_INTER_THREADS,
            intra_threads=settings.WHISPER_INTRA_THREADS,
        )
    return _whisper

@celery_app.task(name="tasks.diarize_full")
def diarize_full(filepath: str):
    pipeline = get_diarizer()
    diarization = pipeline(filepath)
    segments = [segment for _, _, segment in diarization.itertracks(yield_label=True)]
    # передаём оригинальный путь + список сегментов на GPU
    celery_app.send_task(
        "tasks.transcribe_segments",
        args=(filepath, segments),
        queue="preprocess_gpu"
    )
    return []  # либо храните какие-то метаданные

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(filepath: str, segments):
    model = get_whisper()
    results = []
    for start, end in segments:
        seg_file = f"/tmp/seg_{start:.2f}_{end:.2f}.wav"
        # нарезка ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", filepath,
            "-ss", str(start), "-to", str(end),
            "-ar", "16000", "-ac", "1", seg_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text, _ = model.transcribe(
            seg_file,
            beam_size=settings.ALIGN_BEAM_SIZE
        )
        results.append({"start": start, "end": end, "text": text})
        os.remove(seg_file)
    return results