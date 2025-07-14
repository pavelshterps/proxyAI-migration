import json
import logging
import time
from pathlib import Path
from celery.signals import worker_process_init
from redis import Redis

from config.settings import settings
from config.celery import celery_app
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

# Флаги наличия библиотек
_HF_AVAILABLE = False  # faster_whisper
_PN_AVAILABLE = False  # pyannote.audio

try:
    from faster_whisper import WhisperModel, download_model
    _HF_AVAILABLE = True
    logger.info("[INIT] faster_whisper available")
except ImportError as e:
    logger.warning(f"[INIT] faster_whisper not available: {e}")

try:
    from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
    _PN_AVAILABLE = True
    logger.info("[INIT] pyannote.audio available")
except ImportError as e:
    logger.warning(f"[INIT] pyannote.audio not available: {e}")

# Кешированные экземпляры моделей
_whisper_model = None
_vad = None
_clustering_diarizer = None


def get_whisper_model():
    """Ленивая инициализация WhisperModel."""
    if not _HF_AVAILABLE:
        raise RuntimeError("WhisperModel unavailable")
    global _whisper_model
    if _whisper_model is None:
        model_id   = settings.WHISPER_MODEL_PATH
        cache_dir  = settings.HUGGINGFACE_CACHE_DIR
        device     = settings.WHISPER_DEVICE.lower()
        local_only = (device == "cpu")

        logger.info(f"[INIT] WhisperModel init: model={model_id}, device={device}, local_only={local_only}")
        try:
            path = download_model(model_id, cache_dir=cache_dir, local_files_only=local_only)
            logger.info(f"[INIT] Model cached at: {path}")
        except Exception:
            path = model_id
            logger.info(f"[INIT] Will download '{model_id}' online")
        compute = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8").lower()
        if device == "cpu" and compute in ("fp16", "float16"):
            logger.warning("[INIT] FP16 unsupported on CPU, switching to int8")
            compute = "int8"

        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info("[INIT] WhisperModel loaded successfully")
    return _whisper_model


def get_vad():
    """Ленивая инициализация VAD."""
    if not _PN_AVAILABLE:
        raise RuntimeError("VAD unavailable")
    global _vad
    if _vad is None:
        logger.info("[INIT] Loading VAD model...")
        _vad = VoiceActivityDetection.from_pretrained(
            settings.VAD_MODEL_PATH,
            cache_dir=settings.HUGGINGFACE_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("[INIT] VAD model loaded")
    return _vad


def get_clustering_diarizer():
    """Ленивая инициализация SpeakerDiarization."""
    if not _PN_AVAILABLE:
        raise RuntimeError("Diarizer unavailable")
    global _clustering_diarizer
    if _clustering_diarizer is None:
        logger.info("[INIT] Loading Diarizer model...")
        Path(settings.DIARIZER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=settings.DIARIZER_CACHE_DIR,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("[INIT] Diarizer model loaded")
    return _clustering_diarizer


@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    При старте каждого worker'а прогреваем модели на небольшом примере,
    чтобы не холдить первый реальный запрос.
    """
    sample = Path(__file__).parent / "tests/fixtures/sample.wav"
    try:
        if _HF_AVAILABLE:
            logger.info("[WARMUP] Whisper warm-up")
            get_whisper_model().transcribe(
                str(sample),
                **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
                max_initial_timestamp=settings.PREVIEW_LENGTH_S
            )
            logger.info("[WARMUP] Whisper warm-up completed")
        if _PN_AVAILABLE:
            logger.info("[WARMUP] VAD + Diarizer warm-up")
            get_vad().apply({"audio": str(sample)})
            get_clustering_diarizer().apply({"audio": str(sample)})
            logger.info("[WARMUP] VAD + Diarizer warm-up completed")
    except Exception as e:
        logger.warning(f"[WARMUP] Failed: {e!r}")


@celery_app.task(bind=True, name="tasks.download_audio", queue="transcribe_gpu")
def download_audio(self, upload_id: str, correlation_id: str):
    """
    Ничего не делает — здесь можно было бы обрезать видео или получить mp3.
    Для простоты у нас noop.
    """
    logger.info(f"[{correlation_id}] download_audio noop for {upload_id}")


@celery_app.task(bind=True, name="tasks.preview_transcribe", queue="transcribe_gpu")
def preview_transcribe(self, upload_id: str, correlation_id: str):
    """
    Делает транскрипцию первых N секунд (PREVIEW_LENGTH_S) без ffmpeg-обрезки.
    Публикует сразу весь объект preview под ключом "preview".
    """
    if not _HF_AVAILABLE:
        logger.error(f"[{correlation_id}] faster_whisper unavailable — skip preview")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< PREVIEW START >>> for {upload_id}")

    # Конверсия в WAV
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"), None)
    wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    try:
        wav_path = convert_to_wav(src, wav)
        logger.info(f"[{correlation_id}] Converted to WAV: {wav_path}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Conversion error: {e}")
        return

    # Транскрипция первых N секунд
    try:
        model = get_whisper_model()
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        segments, _ = model.transcribe(
            str(wav_path),
            word_timestamps=True,
            max_initial_timestamp=settings.PREVIEW_LENGTH_S,
            **opts
        )
        segs = list(segments)
        logger.info(f"[{correlation_id}] Preview segments: {len(segs)}")
    except Exception as e:
        logger.error(f"[{correlation_id}] Preview error: {e}")
        return

    preview = {
        "text": "".join(s.text for s in segs),
        "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    }

    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[{correlation_id}] Preview saved")

    # Публикуем статус и сам preview
    state = {"status": "preview_done", "preview": preview}
    r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    logger.info(f"[{correlation_id}] <<< PREVIEW DONE >>> in {time.time() - t0:.2f}s")

    # Запускаем полную транскрипцию
    from tasks import transcribe_segments
    transcribe_segments.delay(upload_id, correlation_id)


@celery_app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    """
    Делает полную транскрипцию всего WAV-файла.
    После успешного сохранения посылает статус transcript_done.
    """
    if not _HF_AVAILABLE:
        logger.error(f"[{correlation_id}] faster_whisper unavailable — skip full transcript")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< TRANSCRIBE START >>> for {upload_id}")

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        model = get_whisper_model()
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        segments, _ = model.transcribe(str(wav), word_timestamps=True, **opts)
        segs = list(segments)
        logger.info(f"[{correlation_id}] Full segments count: {len(segs)}")

        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        transcript_file = out_dir / "transcript.json"
        transcript_file.write_text(
            json.dumps([
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segs
            ], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"[{correlation_id}] Transcript JSON saved to {transcript_file}")

        # Публикуем что транскрипция завершена
        state = {"status": "transcript_done", "preview": None}
        r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        logger.info(f"[{correlation_id}] <<< TRANSCRIBE DONE >>> in {time.time() - t0:.2f}s")

        # Автоматический запуск diarization, если пользователь уже просил
        if r.get(f"diarize_requested:{upload_id}") == "1":
            from tasks import diarize_full
            diarize_full.delay(upload_id, correlation_id)

    except Exception as e:
        logger.error(f"[{correlation_id}] Error in full transcription: {e}", exc_info=True)


@celery_app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    """
    Диаризация спикеров по всему файлу.
    Публикует статус diarization_done.
    """
    if not _PN_AVAILABLE:
        logger.error(f"[{correlation_id}] pyannote.audio unavailable — skip diarization")
        return

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< DIARIZE START >>> for {upload_id}")

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        ann = get_clustering_diarizer().apply({"audio": str(wav)})

        segs = [
            {"start": float(seg.start), "end": float(seg.end), "speaker": spk}
            for seg, _, spk in ann.itertracks(yield_label=True)
        ]

        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        diar_file = out_dir / "diarization.json"
        diar_file.write_text(
            json.dumps(segs, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"[{correlation_id}] Diarization JSON saved to {diar_file}")

        # Публикуем что диаризация завершена
        state = {"status": "diarization_done", "preview": None}
        r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        logger.info(f"[{correlation_id}] <<< DIARIZE DONE >>> in {time.time() - t0:.2f}s")

    except Exception as e:
        logger.error(f"[{correlation_id}] Diarization error: {e}", exc_info=True)