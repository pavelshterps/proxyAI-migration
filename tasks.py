import os
import logging
from celery import Celery
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)
app = Celery("proxyai")
app.config_from_object("config")

# Путь для кеша диаризатора (доступен в контейнере)
DIARIZER_CACHE = os.environ.get("DIARIZER_CACHE_DIR", "/tmp/diarizer_cache")

# Глобальные одноразовые объекты
_whisper_model = None
_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = "/hf_cache/models--guillaumekln--faster-whisper-medium"
        logger.info(
            "Loading WhisperModel once at startup: "
            f"{{'model_size_or_path': model_path, 'device': 'cuda', "
            "'compute_type': 'int8', 'device_index': 0}}"
        )
        _whisper_model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="int8",
            device_index=0,
        )
        logger.info("WhisperModel loaded (quantized int8)")
    return _whisper_model

def get_diarizer():
    global _diarizer
    if _diarizer is None:
        os.makedirs(DIARIZER_CACHE, exist_ok=True)
        logger.info(f"Loading Pyannote diarizer with cache at {DIARIZER_CACHE}")
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=True,
            cache_dir=DIARIZER_CACHE,
        )
        logger.info("Pyannote diarizer loaded")
    return _diarizer

@app.task(name="tasks.transcribe_segments")
def transcribe_segments(upload_id: str, segments: list):
    """
    Таск для транскрипции отдельных сегментов аудио.
    :param upload_id: идентификатор загрузки (для логов/хранилища)
    :param segments: список сегментов вида [{'start': ..., 'end': ..., 'path': ...}, ...]
    """
    model = get_whisper_model()
    for segment in segments:
        start, end, path = segment["start"], segment["end"], segment["path"]
        logger.info(f"Transcribing segment {path} [{start}-{end}]")
        # здесь ваша логика вызова model.transcribe(...) и сохранения результата
        # например:
        # result = model.transcribe(path, beam_size=5, language="ru")
        # save_transcript(upload_id, segment, result)

@app.task(name="tasks.diarize_full")
def diarize_full(upload_id: str, audio_path: str):
    """
    Таск для полной диаризации аудио.
    :param upload_id: идентификатор загрузки
    :param audio_path: путь к исходному файлу .wav
    """
    diarizer = get_diarizer()
    logger.info(f"Running diarization for {audio_path}")
    # здесь ваша логика вызова diarizer(audio_path) и разбора спикеров
    # например:
    # diarization = diarizer(audio_path)
    # segments = []
    # for turn, _, speaker in diarization.itertracks(yield_label=True):
    #     segments.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
    # save_diarization(upload_id, segments)