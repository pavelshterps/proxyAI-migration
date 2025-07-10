import os
import json
import logging
import time
from pathlib import Path

from celery import shared_task
from celery.signals import worker_process_init
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from pydub import AudioSegment
from redis import Redis

from config.settings import settings
from config.celery import app                # ← единый экземпляр
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_path = settings.WHISPER_MODEL_PATH
        device     = settings.WHISPER_DEVICE.lower()
        compute    = settings.WHISPER_COMPUTE_TYPE.lower()
        if device == "cpu" and compute in ("float16", "fp16"):
            logger.warning(
                f"Compute type '{compute}' not supported on CPU; falling back to int8"
            )
            compute = "int8"
        logger.info(
            f"Loading WhisperModel (path={model_path}, device={device}, compute={compute})"
        )
        try:
            _whisper_model = WhisperModel(model_path, device=device, compute_type=compute)
        except Exception as e:
            logger.warning(
                f"Failed to load WhisperModel with compute={compute}: {e}; retrying int8"
            )
            _whisper_model = WhisperModel(model_path, device=device, compute_type="int8")
        logger.info("WhisperModel loaded")
    return _whisper_model

def get_vad():
    global _vad
    if _vad is None:
        model_id = getattr(
            settings, "VAD_MODEL_PATH", "pyannote/voice-activity-detection"
        )
        _vad = VoiceActivityDetection.from_pretrained(
            model_id, use_auth_token=settings.HUGGINGFACE_TOKEN
        )
        logger.info("VAD pipeline loaded")
    return _vad

def get_clustering_diarizer():
    global _clustering_diarizer
    if _clustering_diarizer is None:
        cache_dir = settings.DIARIZER_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Loading clustering SpeakerDiarization into cache '{cache_dir}'")
        _clustering_diarizer = SpeakerDiarization.from_pretrained(
            settings.PYANNOTE_PIPELINE,
            cache_dir=cache_dir,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )
        logger.info("Clustering-based diarizer loaded")
    return _clustering_diarizer

@worker_process_init.connect
def preload_and_warmup(**kwargs):
    """
    Warm up models on worker start.
    """
    sample = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample.wav"
    device = settings.WHISPER_DEVICE.lower()

    if device == "cpu":
        try:
            opts = {}
            if settings.WHISPER_LANGUAGE:
                opts["language"] = settings.WHISPER_LANGUAGE
            get_whisper_model().transcribe(str(sample), **opts)
            logger.info("✅ Warm-up WhisperModel complete (CPU worker)")
        except Exception as e:
            logger.warning(f"Warm-up WhisperModel failed: {e}")
    else:
        try:
            get_vad().apply({"audio": str(sample)})
            logger.info("✅ Warm-up VAD complete (GPU worker)")
        except Exception as e:
            logger.warning(f"Warm-up VAD failed: {e}")
        try:
            get_clustering_diarizer().apply({"audio": str(sample)})
            logger.info("✅ Warm-up clustering diarizer complete (GPU worker)")
        except Exception as e:
            logger.warning(f"Warm-up clustering diarizer failed: {e}")

@shared_task(bind=True, name="tasks.transcribe_segments", queue="preprocess_cpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    """
    Whisper transcription (int8) on CPU.
    """
    redis   = Redis.from_url(settings.CELERY_BROKER_URL)
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    start   = time.time()

    # locate source file (any extension)
    upload_dir = Path(settings.UPLOAD_FOLDER)
    candidates = list(upload_dir.glob(f"{upload_id}.*"))
    if not candidates:
        adapter.error(f"No source file for upload_id={upload_id}")
        redis.publish(f"progress:{upload_id}", "error")
        redis.set(f"progress:{upload_id}", "error")
        return
    upload_path = candidates[0]

    whisper = get_whisper_model()
    try:
        adapter.info(f"Starting transcription for '{upload_path.name}'")
        wav_path = upload_dir / f"{upload_id}.wav"

        # convert if needed
        if upload_path.suffix.lower() != ".wav":
            try:
                convert_to_wav(upload_path, wav_path, sample_rate=16000, channels=1)
                src = wav_path
            except Exception as e:
                adapter.error(f"❌ convert_to_wav failed: {e}", exc_info=True)
                src = upload_path
        else:
            src = upload_path

        dst_dir   = Path(settings.RESULTS_FOLDER) / upload_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        audio     = AudioSegment.from_file(str(src))
        transcript = []
        windows   = split_audio_fixed_windows(src, settings.SEGMENT_LENGTH_S)

        for idx, (start, end) in enumerate(windows):
            adapter.debug(f"Segment {idx}: {start:.1f}-{end:.1f}s")
            chunk   = audio[int(start*1000):int(end*1000)]
            tmp_wav = dst_dir / f"{upload_id}_{idx}.wav"
            chunk.export(tmp_wav, format="wav")

            try:
                opts = {"beam_size": settings.WHISPER_BATCH_SIZE, "word_timestamps": True}
                if settings.WHISPER_LANGUAGE:
                    opts["language"] = settings.WHISPER_LANGUAGE
                segments, _ = whisper.transcribe(str(tmp_wav), **opts)
            except Exception as e:
                adapter.error(f"❌ Transcription failed on segment {idx}: {e}", exc_info=True)
                tmp_wav.unlink(missing_ok=True)
                continue

            for seg in segments:
                transcript.append({
                    "segment": idx,
                    "start":   start + seg.start,
                    "end":     start + seg.end,
                    "text":    seg.text,
                })
            tmp_wav.unlink(missing_ok=True)

        out_file = dst_dir / "transcript.json"
        out_file.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        elapsed = time.time() - start
        adapter.info(f"✅ Transcription saved in {elapsed:.2f}s")

        redis.publish(f"progress:{upload_id}", "50%")
        redis.set(f"progress:{upload_id}", "50%")

    except Exception as e:
        adapter.error(f"❌ Fatal error in transcribe_segments: {e}", exc_info=True)
        redis.publish(f"progress:{upload_id}", "error")
        redis.set(f"progress:{upload_id}", "error")
        raise

@shared_task(bind=True, name="tasks.diarize_full", queue="preprocess_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    """
    Pyannote diarization on GPU.
    """
    redis   = Redis.from_url(settings.CELERY_BROKER_URL)
    adapter = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    start   = time.time()

    try:
        adapter.info(f"Starting diarization for upload_id={upload_id}")
        src_wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"

        if not src_wav.exists():
            candidates = list(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
            if not candidates:
                raise FileNotFoundError(f"No source file for {upload_id}")
            orig = candidates[0]
            try:
                convert_to_wav(orig, src_wav, sample_rate=16000, channels=1)
                src = src_wav
            except Exception as e:
                adapter.error(f"❌ convert_to_wav for diarization failed: {e}", exc_info=True)
                src = orig
        else:
            src = src_wav

        speech   = get_vad().apply({"audio": str(src)})
        speakers = []

        if settings.USE_FS_EEND:
            adapter.info("Using FS-EEND diarization")
            pipeline = SpeakerDiarization.from_pretrained(
                settings.FS_EEND_PIPELINE, use_auth_token=settings.HUGGINGFACE_TOKEN
            )
            diar = pipeline({"audio": str(src)})
            for segment, _, spk in diar.itertracks(yield_label=True):
                speakers.append({"start": segment.start, "end": segment.end, "speaker": spk})
        else:
            adapter.info("Using clustering-based diarization")
            ann = get_clustering_diarizer().apply({"audio": str(src)})
            for turn, _, spk in ann.itertracks(yield_label=True):
                speakers.append({"start": turn.start, "end": turn.end, "speaker": spk})

        dst_dir = Path(settings.RESULTS_FOLDER) / upload_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_file = dst_dir / "diarization.json"
        out_file.write_text(
            json.dumps(speakers, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        elapsed = time.time() - start
        adapter.info(f"✅ Diarization saved in {elapsed:.2f}s")

        redis.publish(f"progress:{upload_id}", "100%")
        redis.set(f"progress:{upload_id}", "100%")

    except Exception as e:
        adapter.error(f"❌ Fatal error in diarize_full: {e}", exc_info=True)
        redis.publish(f"progress:{upload_id}", "error")
        redis.set(f"progress:{upload_id}", "error")
        raise

@shared_task(name="tasks.cleanup_old_uploads")
def cleanup_old_uploads():
    """
    Deletes upload files older than 24 hours.
    Scheduled via Celery Beat.
    """
    cutoff     = time.time() - 24 * 3600
    upload_dir = Path(settings.UPLOAD_FOLDER)
    for file_path in upload_dir.iterdir():
        try:
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                logger.info(f"Deleted old upload file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")

def split_audio_fixed_windows(audio_path: Path, window_s: int):
    audio     = AudioSegment.from_file(str(audio_path))
    length_ms = len(audio)
    window_ms = window_s * 1000
    return [
        (start / 1000.0, min(start + window_ms, length_ms) / 1000.0)
        for start in range(0, length_ms, window_ms)
    ]