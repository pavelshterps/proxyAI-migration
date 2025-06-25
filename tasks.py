import os
from celery_app import celery_app
from config.settings import settings
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# Принудительно кешировать HF в том hf_cache
os.environ["HF_HOME"] = "/hf_cache"
os.environ["HUGGINGFACE_HUB_TOKEN"] = settings.HUGGINGFACE_TOKEN

# Один раз подгружаем пайплайн диаризации на CPU
diarizer = Pipeline.from_pretrained(
    settings.PYANNOTE_PROTOCOL,
    use_auth_token=settings.HUGGINGFACE_TOKEN
)

def get_whisper_model():
    return WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.DEVICE,
        device_index=0,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        inter_threads=1,
        intra_threads=1,
        cache_dir="/hf_cache"
    )

@celery_app.task(name="tasks.diarize_full", queue="preprocess_cpu")
def diarize_full(filepath: str):
    # запустить диаризацию целиком
    diarization = diarizer(filepath)
    segments = [
        (segment.start, segment.end)
        for segment, _, _ in diarization.itertracks(yield_label=True)
    ]
    # передать дальше на транскрипцию
    result = celery_app.send_task(
        "tasks.transcribe_segments",
        args=(filepath, segments),
        queue="preprocess_gpu"
    ).get()
    return result

@celery_app.task(name="tasks.transcribe_segments", queue="preprocess_gpu")
def transcribe_segments(filepath: str, segments):
    model = get_whisper_model()
    all_text = []
    for (start, end) in segments:
        segments_result, _ = model.transcribe(
            filepath,
            segment=[start, end],
            beam_size=settings.ALIGN_BEAM_SIZE
        )
        all_text.extend(segments_result)
    return all_text