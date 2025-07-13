import json
import logging
import requests
import subprocess
import time
from pathlib import Path
from celery.signals import worker_process_init
from faster_whisper import WhisperModel, download_model
from pyannote.audio.pipelines import VoiceActivityDetection, SpeakerDiarization
from redis import Redis

from config.settings import settings
from config.celery import celery_app
from utils.audio import convert_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_vad = None
_clustering_diarizer = None

# … (get_whisper_model, get_vad as before) …

def get_clustering_diarizer():
    """Lazy init of Speaker Diarization."""
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

# … (preload_and_warmup, download_audio, preview_transcribe, transcribe_segments) …

@celery_app.task(bind=True, name="tasks.transcribe_segments", queue="transcribe_gpu")
def transcribe_segments(self, upload_id: str, correlation_id: str):
    """
    Full transcription on the entire WAV, write transcript.json, publish SSE.
    """
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< TRANSCRIBE START >>> for {upload_id}")

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        model = get_whisper_model()
        opts = {"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}
        segs, _meta = model.transcribe(str(wav), word_timestamps=True, **opts)
        segs = list(segs)
        logger.info(f"[{correlation_id}] Full segments count: {len(segs)}")

        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = out_dir / "transcript.json"
        transcript_file.write_text(
            json.dumps([{"start":s.start,"end":s.end,"text":s.text} for s in segs],
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"[{correlation_id}] Transcript JSON saved to {transcript_file}")

        state = {"status":"transcript_done","preview":None}
        r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        logger.info(f"[{correlation_id}] <<< TRANSCRIBE DONE >>> for {upload_id} in {time.time()-t0:.2f}s")

        # chain to diarization if requested
        if r.get(f"diarize_requested:{upload_id}") == "1":
            logger.info(f"[{correlation_id}] Chaining diarization for {upload_id}")
            diarize_full.delay(upload_id, correlation_id)
    except Exception as e:
        logger.error(f"[{correlation_id}] Error in full transcription: {e}", exc_info=True)

@celery_app.task(bind=True, name="tasks.diarize_full", queue="diarize_gpu")
def diarize_full(self, upload_id: str, correlation_id: str):
    """
    Speaker diarization, write diarization.json, publish SSE.
    """
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    t0 = time.time()
    logger.info(f"[{correlation_id}] <<< DIARIZE START >>> for {upload_id}")

    try:
        wav = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
        ann = get_clustering_diarizer().apply({"audio": str(wav)})

        segs = []
        for seg, _, spk in ann.itertracks(yield_label=True):
            segs.append({"start": float(seg.start),
                         "end":   float(seg.end),
                         "speaker": spk})

        out_dir = Path(settings.RESULTS_FOLDER) / upload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        diar_file = out_dir / "diarization.json"
        diar_file.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[{correlation_id}] Diarization JSON saved to {diar_file}")

        state = {"status":"diarization_done","preview":None}
        r.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        r.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        logger.info(f"[{correlation_id}] <<< DIARIZE DONE >>> for {upload_id} in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.error(f"[{correlation_id}] Diarization error: {e}", exc_info=True)