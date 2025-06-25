import os
import tempfile
import ffmpeg
from celery_app import celery_app
from config.settings import settings
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# Диаризация на весь файл (CPU)
diarizer = Pipeline.from_pretrained(
    settings.PYANNOTE_PROTOCOL,
    use_auth_token=settings.HUGGINGFACE_TOKEN,
)

@celery_app.task(name="tasks.diarize_full", queue="preprocess_cpu")
def diarize_full(filepath: str):
    diarization = diarizer(filepath)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    # отсылаем каждый сегмент на GPU-транскрибуцию
    for seg in segments:
        transcribe_segments.apply_async(
            args=(filepath, seg["start"], seg["end"], seg["speaker"])
        )
    return segments

# Ленивая инициализация WhisperModel
_whisper_model = None
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=0,
            inter_threads=1,
            intra_threads=1,
        )
    return _whisper_model

# Транскрипция отдельных сегментов (GPU)
@celery_app.task(name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(filepath: str, start: float, end: float, speaker: str):
    model = get_whisper_model()
    # вырезаем сегмент во временный файл
    snippet = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    (
        ffmpeg
        .input(filepath, ss=start, to=end)
        .output(snippet, format="wav", acodec="pcm_s16le")
        .run(quiet=True, overwrite_output=True)
    )
    result, _ = model.transcribe(
        snippet,
        beam_size=settings.ALIGN_BEAM_SIZE,
    )
    # пофильтруем и добавим спикера в каждый кусочек
    for r in result:
        r["speaker"] = speaker
    os.remove(snippet)
    return result