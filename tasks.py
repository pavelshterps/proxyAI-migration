import os, tempfile
from celery_app import celery_app
from config.settings import settings
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# инициализируем диаризатор ОДИН раз, чтобы не скачивать каждый вызов
diarizer = Pipeline.from_pretrained(
    settings.PYANNOTE_PROTOCOL,
    use_auth_token=settings.HUGGINGFACE_TOKEN
)

# кеш для модели Whisper
_model = None
def get_whisper_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            device_index=0,
            inter_threads=1,
            intra_threads=1,
            cache_dir=os.getenv("HF_CACHE_DIR", "/hf_cache")
        )
    return _model

@celery_app.task(
    name="tasks.diarize_full",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3}
)
def diarize_full(self, filepath):
    job_id = self.request.id
    segments = []
    # 1) диаризация по всему файлу
    diar = diarizer(filepath)
    for turn, _, speaker in diar.itertracks(yield_label=True):
        # вырезаем отрезки ffmpeg-ом
        seg_path = os.path.join(
            settings.UPLOAD_FOLDER,
            f"{job_id}_{turn.start:.2f}_{turn.end:.2f}.wav"
        )
        os.system(
            f"ffmpeg -y -i {filepath} -ss {turn.start} -to {turn.end} -c copy {seg_path}"
        )
        segments.append((seg_path, speaker))
    # 2) транскрибируем каждый сегмент
    results = []
    for seg_path, speaker in segments:
        res = transcribe_segment.delay(seg_path, speaker)
        results.append(res)
    # 3) ждём и собираем
    full = []
    for r in results:
        full.append(r.get(timeout=3600))
    return full

@celery_app.task(name="tasks.transcribe_segment")
def transcribe_segment(filepath, speaker):
    model = get_whisper_model()
    segments, _ = model.transcribe(
        filepath,
        beam_size=settings.ALIGN_BEAM_SIZE
    )
    return {
        "speaker": speaker,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ]
    }