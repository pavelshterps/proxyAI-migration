import os
from celery_app import celery_app
from config.settings import settings

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

_whisper_model = None
_diarizer = None

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        # читаем переменную из окружения, примонтированную из docker-compose
        cache_dir = os.getenv("HF_CACHE_DIR", "/hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        _diarizer = Pipeline.from_pretrained(
            settings.PYANNOTE_PROTOCOL,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
    return _diarizer

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.DEVICE,
            device_index=0,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            inter_threads=1,
            intra_threads=1
        )
    return _whisper_model

@celery_app.task(
    name="tasks.diarize_full",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
)
def diarize_full(self, filepath):
    job_id = self.request.id
    diar = get_diarizer()
    diarization = diar(filepath)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        seg_path = os.path.join(
            settings.UPLOAD_FOLDER,
            f"{job_id}_{start:.2f}_{end:.2f}.wav"
        )
        # ffmpeg или pydub, по вкусу
        os.system(f"ffmpeg -y -i {filepath} -ss {start} -to {end} -c copy {seg_path}")
        segments.append((seg_path, speaker))

    # отправляем сегментные задачи
    results = []
    for seg_path, speaker in segments:
        results.append(transcribe_segment.delay(seg_path, speaker))

    # ждём их завершения
    output = [r.get(timeout=3600) for r in results]
    return output

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