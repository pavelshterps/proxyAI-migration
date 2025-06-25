import os
from celery_app import celery_app
from config.settings import settings

# ленивые модули
diarizer = None
whisper_model = None

def get_diarizer():
    global diarizer
    if diarizer is None:
        from pyannote.audio import Pipeline
        os.environ["HF_HOME"] = settings.HF_CACHE
        diarizer = Pipeline.from_pretrained(settings.PYANNOTE_MODEL)
    return diarizer

def get_whisper():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        os.environ["HF_HOME"] = settings.HF_CACHE
        whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE_INDEX,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            inter_threads=settings.WHISPER_INTER_THREADS,
            intra_threads=settings.WHISPER_INTRA_THREADS,
        )
    return whisper_model

@celery_app.task(name="tasks.diarize_full")
def diarize_full(filepath: str):
    pipeline = get_diarizer()
    diarization = pipeline(filepath)
    # соберём отрезки для передачи в сегментацию
    segments = [segment for turn, _, segment in diarization.itertracks(yield_label=True)]
    # запускаем сегментацию на GPU
    celery_app.send_task("tasks.transcribe_segments", args=(filepath, segments), queue="preprocess_gpu")
    return []  # пока пусто

@celery_app.task(name="tasks.transcribe_segments")
def transcribe_segments(filepath: str, segments):
    model = get_whisper()
    results = []
    for start, end in segments:
        segment_file = f"/tmp/segment_{start:.2f}_{end:.2f}.wav"
        # извлечение сегмента (ffmpeg или библиотека)
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", filepath,
            "-ss", str(start), "-to", str(end),
            "-ar", "16k", "-ac", "1", segment_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text, _ = model.transcribe(segment_file, beam_size=int(settings.ALIGN_BEAM_SIZE) if hasattr(settings, "ALIGN_BEAM_SIZE") else 5)
        results.append({"start": start, "end": end, "text": text})
        os.remove(segment_file)
    return results