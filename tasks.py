import os
import json
import logging
import time
from pathlib import Path

import torch
from celery import Celery, shared_task
from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from utils.audio import convert_to_wav
from pydub import AudioSegment
from redis import Redis

# Импорт FS-EEND
from eend.inference import Inference as EENDInference

from config.settings import settings

app = Celery(
    "proxyai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
# по умолчанию задачи берутся с CPU-очереди, но в декораторах можно указать preprocess_gpu
app.conf.task_default_queue = "preprocess_cpu"

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_diarizer = None
_eend_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device = settings.WHISPER_DEVICE.lower()
        compute = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(
                f"Compute type '{compute}' not supported on CPU; falling back to 'int8'"
            )
            compute = "int8"
        logger.info(
            f"Loading WhisperModel {{'path': '{model_path}', 'device': '{device}', 'compute': '{compute}'}}"
        )
        try:
            _whisper_model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute
            )
        except Exception as e:
            logger.warning(
                f"Failed to init WhisperModel with compute_type={compute}: {e}; retrying with 'int8'"
            )
            _whisper_model = WhisperModel(
                model_path,
                device=device,
                compute_type="int8"
            )
        logger.info("WhisperModel loaded")
    return _whisper_model


def get_vad():
    global _vad
    if _vad is None:
        _vad = VoiceActivityDetection.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("VAD pipeline loaded")
    return _vad


def get_clustering_diarizer():
    global _diarizer
    if _diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading pyannote SpeakerDiarization into cache '{cache_dir}'")
        _diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        pd = getattr(settings, "PYANNOTE_DEVICE", "cpu").lower()
        if pd != "cpu":
            try:
                dev = torch.device(pd)
                _diarizer.to(dev)
                logger.info(f"Diarizer moved to {dev}")
            except Exception as e:
                logger.warning(f"Could not move diarizer to {pd}: {e}")
        logger.info("Clustering-based diarizer loaded")
    return _diarizer


def get_eend_model():
    global _eend_model
    if _eend_model is None:
        _eend_model = EENDInference(
            model=settings.FS_EEND_MODEL_PATH,
            device=settings.FS_EEND_DEVICE
        )
        logger.info("FS-EEND model loaded")
    return _eend_model


@worker_process_init.connect
def preload_and_warmup(**kwargs):
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    # тёплая прогонка: VAD
    try:
        get_vad().apply({"audio": str(sample)})
        logger.info("✅ Warm-up VAD complete")
    except Exception:
        pass
    # тёплая прогонка: clustering diarizer
    try:
        get_clustering_diarizer().apply({"audio": str(sample)})
        logger.info("✅ Warm-up clustering diarizer complete")
    except Exception:
        pass
    # тёплая прогонка: FS-EEND
    if settings.USE_FS_EEND and settings.FS_EEND_MODEL_PATH:
        try:
            get_eend_model().diarize(str(sample))
            logger.info("✅ Warm-up FS-EEND complete")
        except Exception:
            pass
    # тёплая прогонка: Whisper
    try:
        opts = {}
        if settings.WHISPER_LANGUAGE:
            opts["language"] = settings.WHISPER_LANGUAGE
        get_whisper_model().transcribe(str(sample), **opts)
        logger.info("✅ Warm-up WhisperModel complete")
    except Exception:
        pass


@shared_task(
    bind=True,
    name="tasks.transcribe_segments",
    queue="preprocess_gpu"
)
def transcribe_segments(self, upload_id: str, correlation_id: str):
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    whisper = get_whisper_model()

    # измеряем задержку в очереди
    headers = getattr(self.request, "headers", {}) or {}
    enqueue_header = headers.get("enqueue_time")
    if enqueue_header:
        try:
            delta = time.time() - float(enqueue_header)
            adapter.info(f"⏳ Queue delay: {delta:.2f}s")
        except ValueError:
            adapter.warning(f"Invalid enqueue_time header: {enqueue_header}")
    task_start = time.time()

    # конвертация в WAV, если нужно
    src_orig = Path(settings.UPLOAD_FOLDER) / upload_id
    wav_name = Path(upload_id).stem + ".wav"
    wav_src = Path(settings.UPLOAD_FOLDER) / wav_name
    if src_orig.suffix.lower() != ".wav":
        convert_to_wav(src_orig, wav_src, sample_rate=16000, channels=1)
        src = wav_src
    else:
        src = src_orig

    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    adapter.info(f"Starting transcription for '{src}'")
    full_audio = AudioSegment.from_file(str(src))
    transcript = []
    windows = split_audio_fixed_windows(src, settings.SEGMENT_LENGTH_S)

    for idx, (start, end) in enumerate(windows):
        adapter.debug(f" Transcribing segment {idx}: {start:.1f}s → {end:.1f}s")
        chunk = full_audio[int(start * 1000):int(end * 1000)]
        tmp_wav = dst_dir / f"{upload_id}_{idx}.wav"
        chunk.export(tmp_wav, format="wav")

        try:
            opts = {
                "beam_size": settings.WHISPER_BATCH_SIZE,
                "word_timestamps": True,
            }
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE

            segments, _ = whisper.transcribe(str(tmp_wav), **opts)
        except Exception as e:
            adapter.error(f"❌ Transcription failed on segment {idx}: {e}", exc_info=True)
            tmp_wav.unlink()
            continue

        for seg in segments:
            transcript.append({
                "segment": idx,
                "start":   start + seg.start,
                "end":     start + seg.end,
                "text":    seg.text
            })
        tmp_wav.unlink()

    out_path = dst_dir / "transcript.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    adapter.info(f"Transcription complete: saved to '{out_path}' ({time.time() - task_start:.2f}s)")

    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    redis.publish(f"progress:{upload_id}", "50%")
    redis.set(f"progress:{upload_id}", "50%")


@shared_task(
    bind=True,
    name="tasks.diarize_full",
    queue="preprocess_gpu"
)
def diarize_full(self, upload_id: str, correlation_id: str):
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    task_start = time.time()

    # путь до WAV
    wav_name = Path(upload_id).stem + ".wav"
    src = Path(settings.UPLOAD_FOLDER) / wav_name

    # VAD предобработка
    vad = get_vad()
    speech = vad.apply({"audio": str(src)})
    regions = speech.get_timeline().support()

    # выбор алгоритма
    speakers = []
    if settings.USE_FS_EEND and settings.FS_EEND_MODEL_PATH:
        adapter.info("Using FS-EEND diarization")
        eend = get_eend_model()
        labels = eend.diarize(str(src))
        frame_shift = settings.FRAME_SHIFT
        for frame_idx, row in enumerate(labels):
            t0 = frame_idx * frame_shift
            for spk_idx, active in enumerate(row):
                if active:
                    speakers.append({"start": t0, "end": t0 + frame_shift, "speaker": spk_idx})
    else:
        adapter.info("Using clustering-based pyannote diarization")
        diar = get_clustering_diarizer()
        ann = diar.apply({"audio": str(src)})
        for turn, _, spk in ann.itertracks(yield_label=True):
            speakers.append({"start": turn.start, "end": turn.end, "speaker": spk})

    # сохраняем результат
    dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / "diarization.json"
    out_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2))
    adapter.info(f"Diarization complete: saved to '{out_path}' ({time.time() - task_start:.2f}s)")

    redis = Redis.from_url(settings.CELERY_BROKER_URL)
    redis.publish(f"progress:{upload_id}", "100%")
    redis.set(f"progress:{upload_id}", "100%")


def split_audio_fixed_windows(audio_path: Path, window_s: int):
    audio = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    return [
        (start_ms / 1000.0, min(start_ms + window_ms, length_ms) / 1000.0)
        for start_ms in range(0, length_ms, window_ms)
    ]