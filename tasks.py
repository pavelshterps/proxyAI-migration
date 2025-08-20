import json
import logging
import subprocess
import time
import re
import math
import os
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import warnings
import logging as _logging
import requests
from redis import Redis
from celery.signals import worker_process_init

from celery_app import app  # импорт Celery instance
from config.settings import settings

# --- Logger setup ---
logger = logging.getLogger("tasks")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# скрыть массивные DeprecationWarning от torchaudio/Lightning
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends has been deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*deprecated.*")
# можно понизить болтливость Lightning
_logging.getLogger("lightning.pytorch.utilities.upgrade_checkpoint").setLevel(_logging.ERROR)
_logging.getLogger("pytorch_lightning.utilities.upgrade_checkpoint").setLevel(_logging.ERROR)

# --- Model availability flags & holders ---
_HF_AVAILABLE = False
_PN_AVAILABLE = False
_whisper_model = None
_diarization_pipeline = None

# --- Speaker stitching / embedding ---
SPEAKER_STITCH_ENABLED = getattr(settings, "SPEAKER_STITCH_ENABLED", True)
SPEAKER_STITCH_THRESHOLD = float(getattr(settings, "SPEAKER_STITCH_THRESHOLD", 0.85))
SPEAKER_STITCH_POOL_SIZE = int(getattr(settings, "SPEAKER_STITCH_POOL_SIZE", 8))
SPEAKER_STITCH_EMA_ALPHA = float(getattr(settings, "SPEAKER_STITCH_EMA_ALPHA", 0.5))
SPEAKER_STITCH_MERGE_THRESHOLD = float(getattr(settings, "SPEAKER_STITCH_MERGE_THRESHOLD", 0.98))
TRANSCRIPTION_CHUNK_PAD_S = float(getattr(settings, "TRANSCRIPTION_CHUNK_PAD_S", 10.0))
_speaker_embedding_model = None  # type: ignore

try:
    from faster_whisper import WhisperModel, download_model

    _HF_AVAILABLE = True
    logger.info("[INIT] faster-whisper available")
except ImportError as e:
    logger.warning(f"[INIT] faster-whisper not available: {e}")

try:
    from pyannote.audio import Pipeline as PyannotePipeline

    _PN_AVAILABLE = True
    logger.info("[INIT] pyannote.audio available")
except ImportError as e:
    logger.warning(f"[INIT] pyannote.audio not available: {e}")

# ---------------------- Helpers ----------------------

def send_webhook_event(event_type: str, upload_id: str, data: Optional[Any]):
    url = settings.WEBHOOK_URL
    secret = settings.WEBHOOK_SECRET
    if not url or not secret:
        return

    payload = {
        "event_type": event_type,
        "upload_id": upload_id,
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "data": data,
    }
    headers = {"Content-Type": "application/json", "X-WebHook-Secret": secret}

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=(5, 30))
        except requests.RequestException as e:
            logger.warning(
                f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} network error "
                f"(attempt {attempt}/{max_attempts}) for {upload_id}: {e}"
            )
        else:
            code = resp.status_code
            if 200 <= code < 300 or code == 405:
                logger.info(
                    f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} "
                    f"{'treated as success' if code == 405 else 'succeeded'} "
                    f"(attempt {attempt}/{max_attempts}) for {upload_id}"
                )
                return
            if 400 <= code < 500:
                logger.error(
                    f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} returned {code} "
                    f"for {upload_id}, aborting"
                )
                return
            logger.warning(
                f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} returned {code} "
                f"(attempt {attempt}/{max_attempts}), retrying"
            )
        if attempt < max_attempts:
            time.sleep(30)

    logger.error(
        f"[{datetime.utcnow().isoformat()}] [WEBHOOK] {event_type} failed after "
        f"{max_attempts} attempts for {upload_id}"
    )

# ---- helper для корректного подсчёта занятости GPU ----
def _count_gpu_heavy_tasks(inspector, exclude_task_id: str = None) -> int:
    active = inspector.active() or {}
    heavy_gpu = 0
    for worker, tasks in active.items():
        for t in tasks:
            # не считаем самих себя
            if exclude_task_id and t.get("id") == exclude_task_id:
                continue
            # считаем только задачи, пришедшие в GPU-очереди
            rk = ((t.get("delivery_info") or {}).get("routing_key") or "")
            if rk not in ("transcribe_gpu", "diarize_gpu"):
                continue
            if t.get("name") in ("tasks.diarize_full", "tasks.transcribe_segments"):
                heavy_gpu += 1
    return heavy_gpu


def probe_audio(src: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)],
        capture_output=True,
        text=True
    )
    info = {"duration": 0.0}
    try:
        j = json.loads(res.stdout)
        info["duration"] = float(j["format"].get("duration", 0.0))
        for s in j.get("streams", []):
            if s.get("codec_type") == "audio":
                info.update({
                    "codec_name": s.get("codec_name"),
                    "sample_rate": int(s.get("sample_rate", 0)),
                    "channels": int(s.get("channels", 0)),
                })
                break
    except Exception:
        pass
    return info

# === audio filters: управляем шумодавом/нормализацией/гейном через settings ===

AUDIO_ENABLE_HIGHPASS = bool(getattr(settings, "AUDIO_ENABLE_HIGHPASS", False))
AUDIO_HP_F = int(getattr(settings, "AUDIO_HP_F", 120))  # Hz

AUDIO_ENABLE_DENOISE = bool(getattr(settings, "AUDIO_ENABLE_DENOISE", True))
# afftdn сила подавления шума: 0..64 (в ffmpeg), 6..24 — мягко; повышай для сильнее
AUDIO_AFFTDN_NR = int(getattr(settings, "AUDIO_AFFTDN_NR", 6))

AUDIO_ENABLE_LOUDNORM = bool(getattr(settings, "AUDIO_ENABLE_LOUDNORM", True))
# Параметры EBU R128 loudnorm
AUDIO_LOUDNORM = getattr(settings, "AUDIO_LOUDNORM", "I=-23:TP=-2:LRA=11")

# Доп. вариант: чистый гейн без нормализации/шумодава (для экспериментов)
AUDIO_ENABLE_GAIN = bool(getattr(settings, "AUDIO_ENABLE_GAIN", False))
AUDIO_GAIN_DB = float(getattr(settings, "AUDIO_GAIN_DB", 6.0))  # +6 dB по умолчанию
# --- Whisper decoding/timing flags ---
WHISPER_WORD_TIMESTAMPS = bool(getattr(settings, "WHISPER_WORD_TIMESTAMPS", False))
WHISPER_BEAM_SIZE = int(getattr(settings, "WHISPER_BEAM_SIZE", 1))
WHISPER_BEST_OF = int(getattr(settings, "WHISPER_BEST_OF", 1))
WHISPER_TEMPERATURE = float(getattr(settings, "WHISPER_TEMPERATURE", 0.0))
WHISPER_CONDITION_ON_PREV_TEXT = bool(getattr(settings, "WHISPER_CONDITION_ON_PREV_TEXT", True))

def _ffmpeg_filter_chain() -> str:
    """
    Собираем -af фильтры динамически по флагам settings:
      - AUDIO_ENABLE_HIGHPASS (highpass=f=...)
      - AUDIO_ENABLE_DENOISE (afftdn=nr=...)
      - AUDIO_ENABLE_LOUDNORM (loudnorm=...)
      - AUDIO_ENABLE_GAIN (volume=...dB) — обычно ВМЕСТО loudnorm, но можем и вместе.
    """
    chain = []
    if AUDIO_ENABLE_HIGHPASS:
        chain.append(f"highpass=f={AUDIO_HP_F}")
    if AUDIO_ENABLE_DENOISE:
        chain.append(f"afftdn=nr={AUDIO_AFFTDN_NR}")
    if AUDIO_ENABLE_LOUDNORM:
        chain.append(f"loudnorm={AUDIO_LOUDNORM}")
    if AUDIO_ENABLE_GAIN:
        chain.append(f"volume={AUDIO_GAIN_DB}dB")
    return ",".join(chain)


def prepare_wav(upload_id: str) -> (Path, float):
    """
    Готовит / нормализует WAV 16kHz mono PCM.
    Экспериментальные режимы включаются флагами:
      - AUDIO_ENABLE_HIGHPASS      (True/False)
      - AUDIO_ENABLE_DENOISE       (True/False)
      - AUDIO_ENABLE_LOUDNORM      (True/False)
      - AUDIO_ENABLE_GAIN (+ AUDIO_GAIN_DB) — просто сделать громче.
    """
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    tmp_out = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.tmp.wav"

    info = probe_audio(src)
    duration = float(info.get("duration", 0.0) or 0.0)

    if (
        src.suffix.lower() == ".wav"
        and info.get("codec_name") == "pcm_s16le"
        and int(info.get("sample_rate", 0)) == 16000
        and int(info.get("channels", 0)) == 1
    ):
        if src != target:
            src.rename(target)
        return target, duration

    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass

    ffmpeg_base = [
        "ffmpeg", "-y",
        "-threads", str(settings.FFMPEG_THREADS),
        "-hide_banner", "-nostdin",
        "-i", str(src),
        "-vn",
    ]

    # 1) Пытаемся с фильтрами из текущего профиля
    filters = _ffmpeg_filter_chain()
    try:
        cmd = ffmpeg_base + (["-af", filters] if filters else []) + [
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            str(tmp_out),
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(f"[{upload_id}] ffmpeg OK with filters: {filters or '(none)'}")
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"[{upload_id}] ffmpeg with filters failed: {e}. "
            f"stderr: {e.stderr[:3000] if e.stderr else 'no-stderr'}. Falling back to plain resample."
        )
        # 2) fallback — без фильтров
        try:
            cmd2 = ffmpeg_base + [
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                str(tmp_out),
            ]
            subprocess.run(
                cmd2,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.info(f"[{upload_id}] ffmpeg plain resample OK")
        except subprocess.CalledProcessError as e2:
            logger.error(
                f"[{upload_id}] ffmpeg plain resample failed. "
                f"stderr: {e2.stderr[:3000] if e2.stderr else 'no-stderr'}"
            )
            raise

    try:
        if target.exists():
            target.unlink()
    except Exception:
        pass
    tmp_out.replace(target)

    try:
        info2 = probe_audio(target)
        duration = float(info2.get("duration", duration) or duration)
    except Exception:
        pass

    return target, duration

def prepare_preview_segment(upload_id: str) -> subprocess.Popen:
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    return subprocess.Popen([
        "ffmpeg", "-y", "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
        "-ss", "0", "-t", str(settings.PREVIEW_LENGTH_S),
        "-i", str(src),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        "-f", "wav", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def group_into_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    SILENCE_GAP_S = getattr(settings, "SENTENCE_MAX_GAP_S", 0.35)
    MAX_WORDS = getattr(settings, "SENTENCE_MAX_WORDS", 50)

    sentences = []
    buf = {"start": None, "end": None, "speaker": None, "text": []}
    sentence_end_re = re.compile(r"[\.!\?]$")

    def flush_buffer():
        if buf["text"] and buf["start"] is not None:
            sentences.append({
                "start": buf["start"],
                "end": buf["end"],
                "speaker": buf["speaker"],
                "text": " ".join(buf["text"]),
            })
        buf["start"] = buf["end"] = buf["speaker"] = None
        buf["text"] = []

    for seg in segments:
        txt = seg["text"].strip()
        if not txt:
            continue

        if buf["start"] is None:
            buf["start"] = seg["start"]
            buf["speaker"] = seg.get("speaker")

        if buf["end"] is not None and (seg["start"] - buf["end"] > SILENCE_GAP_S):
            flush_buffer()
            buf["start"] = seg["start"]
            buf["speaker"] = seg.get("speaker")

        buf["end"] = seg["end"]
        buf["text"].append(txt)

        word_count = sum(len(t.split()) for t in buf["text"])
        if sentence_end_re.search(txt) or word_count >= MAX_WORDS:
            flush_buffer()

    flush_buffer()
    return sentences

def merge_speakers(
    transcript: List[Dict[str, Any]],
    diar: List[Dict[str, Any]],
    pad: float = 0.2,
) -> List[Dict[str, Any]]:
    if not diar:
        return [{**t, "speaker": None} for t in transcript]

    diar = sorted(diar, key=lambda d: d["start"])
    transcript = sorted(transcript, key=lambda t: t["start"])
    starts = [d["start"] for d in diar]

    from bisect import bisect_left

    def nearest(idx: int, t0: float, t1: float):
        if idx <= 0:
            return diar[0]
        if idx >= len(diar):
            return diar[-1]
        b, a = diar[idx - 1], diar[idx]
        db = max(0.0, t0 - b["end"])
        da = max(0.0, a["start"] - t1)
        return b if db <= da else a

    out = []
    for t in transcript:
        t0 = max(0.0, t["start"] - pad)
        t1 = t["end"] + pad
        i = bisect_left(starts, t1)
        cands = [
            d for d in diar[max(0, i - 8): i + 8]
            if not (d["end"] <= t0 or d["start"] >= t1)
        ]
        best = max(cands, key=lambda d: max(0.0, min(d["end"], t1) - max(d["start"], t0))) \
            if cands else nearest(i, t0, t1)
        out.append({**t, "speaker": best["speaker"]})
    return out

def get_whisper_model(model_override: str = None):
    global _whisper_model
    device = settings.WHISPER_DEVICE.lower()
    compute = getattr(
        settings,
        "WHISPER_COMPUTE_TYPE",
        "float16" if device.startswith("cuda") else "int8",
    ).lower()
    if model_override:
        logger.info(f"[WHISPER] loading override model {model_override}")
        return WhisperModel(model_override, device=device, compute_type=compute)
    if _whisper_model is None:
        model_id = settings.WHISPER_MODEL_PATH
        try:
            path = download_model(
                model_id,
                cache_dir=settings.HUGGINGFACE_CACHE_DIR,
                local_files_only=(device == "cpu"),
            )
        except Exception:
            path = model_id
        if device == "cpu" and compute in ("fp16", "float16"):
            compute = "int8"
        _whisper_model = WhisperModel(path, device=device, compute_type=compute)
        logger.info(f"[WHISPER] loaded model from {path} on {device} with compute_type={compute}")
    return _whisper_model

def get_diarization_pipeline():
    """
    Загружаем pyannote Pipeline и переводим на нужное устройство через .to(...).
    Если активен DiariZen и в sys.path попался его форк pyannote-audio,
    может прилететь AttributeError(PyanNet.load_from_checkpoint). В этом случае
    не падаем, а логгируем и поднимаем осмысленную ошибку, чтобы диаризация
    попробовала DiariZen/остальные пути.
    """
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    model_id = getattr(settings, "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1")
    try:
        pipeline = PyannotePipeline.from_pretrained(
            model_id,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.DIARIZER_CACHE_DIR,
        )
        # перенос на нужное устройство
        target = getattr(settings, "DIARIZER_DEVICE", None)
        if target is None:
            target = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            pipeline.to(torch.device(target))  # у Pipeline есть .to(...)
        except Exception as e:
            logger.warning(f"[DIARIZE] could not move pipeline to device: {e}")

        _diarization_pipeline = pipeline
        logger.info(f"[DIARIZE] loaded pipeline {model_id} and moved to device={target}")
        return _diarization_pipeline

    except AttributeError as e:
        # ключевой признак конфликта с форком DiariZen
        if "PyanNet" in str(e) and "load_from_checkpoint" in str(e):
            logger.error(
                "[DIARIZE] pyannote preload failed due to DiariZen's vendored pyannote-audio "
                "(PyanNet.load_from_checkpoint missing). Will rely on DiariZen backend and "
                "only try pyannote as a last-chance fallback in a clean env."
            )
        else:
            logger.error(f"[DIARIZE] pyannote pipeline load AttributeError: {e}")
        # пробрасываем дальше как «нет доступного pyannote»
        raise RuntimeError("pyannote pipeline unavailable in current env") from e

    except Exception as e:
        logger.error(f"[DIARIZE] pyannote pipeline load failed: {e}", exc_info=True)
        raise

def get_speaker_embedding_model():
    global _speaker_embedding_model
    if _speaker_embedding_model is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            logger.warning(f"[STITCH] speechbrain not available, cannot do speaker stitching: {e}")
            raise
        savedir = Path(settings.DIARIZER_CACHE_DIR) / "spkrec-ecapa-voxceleb"
        _speaker_embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir)
        )
        logger.info("[STITCH] loaded speaker embedding model from speechbrain/spkrec-ecapa-voxceleb")
    return _speaker_embedding_model

def stitch_speakers(raw: List[Dict[str, Any]], wav: Path, upload_id: str) -> List[Dict[str, Any]]:
    """
    Ускоренная склейка спикеров между чанками:
    - считаем эмбеддинг не для каждого сегмента (сэмплирование),
    - используем центральное окно фиксированной длины для эмбеддинга,
    - поддерживаем mapping исходной метки (pyannote) → канонической без повторного эмбеддинга.
    """
    if not SPEAKER_STITCH_ENABLED:
        return raw

    unique_orig = set(seg.get("speaker") for seg in raw)
    if len(unique_orig) <= 1:
        logger.debug(f"[{upload_id}] only one original speaker {unique_orig}, skipping stitching")
        return raw

    # --- параметры (можно вынести в settings) ---
    MIN_SEG_FOR_EMB = float(getattr(settings, "STITCH_MIN_SEG_FOR_EMB_S", 1.2))  # считать эмбеддинг, если сегмент >= 1.2 s
    EMB_EVERY_N = int(getattr(settings, "STITCH_EMB_EVERY_N", 2))               # эмбеддим каждый N-й сегмент
    EMB_WINDOW_S = float(getattr(settings, "STITCH_EMB_WINDOW_S", 2.4))         # длина окна для эмбеддинга
    MAX_EMB_PER_ORIG = int(getattr(settings, "STITCH_MAX_EMB_PER_ORIG", 2000))  # safeguard

    try:
        import torch
        import torchaudio
        import torch.nn.functional as F

        model = get_speaker_embedding_model()

        waveform, sr = torchaudio.load(str(wav))
        if sr != 16000:
            from torchaudio.transforms import Resample
            resampler = Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # моно
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # хранение центроидов и истории
        stitch_centroids: Dict[str, torch.Tensor] = {}
        stitch_histories: Dict[str, List[torch.Tensor]] = {}
        # map из "исходная метка pyannote" → "каноническая"
        orig2canon: Dict[str, str] = {}
        # счетчик эмбеддингов на исходную метку (safeguard)
        orig_emb_count: Dict[str, int] = {}

        next_label_idx = 0
        def new_canonical_label():
            nonlocal next_label_idx
            label = f"spk_{next_label_idx}"
            next_label_idx += 1
            return label

        def _segment_crop_center(start: float, end: float) -> torch.Tensor:
            """Возвращает центральное окно фиксированной длительности для эмбеддинга."""
            dur = end - start
            if dur <= 0:
                return waveform[:, 0:0]
            win = EMB_WINDOW_S if dur >= EMB_WINDOW_S else dur
            mid = (start + end) * 0.5
            s = max(0.0, mid - win * 0.5)
            e = min(float(waveform.size(1) / sr), mid + win * 0.5)
            start_sample = int(s * sr)
            end_sample = int(e * sr)
            if end_sample <= start_sample:
                end_sample = min(start_sample + int(0.2 * sr), waveform.size(1))
            return waveform[:, start_sample:end_sample]

        def _embed(wav_chunk: torch.Tensor) -> Optional[torch.Tensor]:
            if wav_chunk.numel() == 0:
                return None
            with torch.no_grad():
                emb = model.encode_batch(wav_chunk)  # [1, D] или [D]
            emb = emb.squeeze()
            if emb.ndim > 1:
                emb = emb.flatten()
            emb = F.normalize(emb, p=2, dim=0)
            return emb

        stitched: List[Dict[str, Any]] = []
        raw_sorted = sorted(raw, key=lambda x: x["start"])

        for idx, seg in enumerate(raw_sorted):
            start, end = float(seg["start"]), float(seg["end"])
            if end <= start:
                stitched.append(seg)
                continue

            orig_label = seg.get("speaker")
            seg_dur = end - start

            # Если у этой исходной метки уже есть каноническая — можно проставить её без эмбеддинга
            fast_mapped = False
            if orig_label in orig2canon and (idx % EMB_EVERY_N != 0 or seg_dur < MIN_SEG_FOR_EMB):
                seg["speaker"] = orig2canon[orig_label]
                stitched.append(seg)
                fast_mapped = True

            if fast_mapped:
                continue

            # Safeguard: ограничим число эмбеддингов на одну исходную метку
            if orig_label is not None:
                cnt = orig_emb_count.get(orig_label, 0)
                if cnt >= MAX_EMB_PER_ORIG and orig_label in orig2canon:
                    seg["speaker"] = orig2canon[orig_label]
                    stitched.append(seg)
                    continue

            # Кусок для эмбеддинга (центральное окно)
            chunk = _segment_crop_center(start, end)
            if chunk.numel() == 0:
                # не удалось — если есть маппинг, используем его
                if orig_label in orig2canon:
                    seg["speaker"] = orig2canon[orig_label]
                stitched.append(seg)
                continue

            emb = _embed(chunk)
            if emb is None:
                if orig_label in orig2canon:
                    seg["speaker"] = orig2canon[orig_label]
                stitched.append(seg)
                continue

            # Поиск ближайшего канонического центроида
            assigned_label = None
            best_sim = -1.0
            for canon_label, centroid in stitch_centroids.items():
                sim = torch.dot(emb, centroid).item()
                if sim > best_sim:
                    best_sim = sim
                    assigned_label = canon_label

            # Решение: присоединяемся к существующему центроиду или создаём новый
            if assigned_label is not None and best_sim >= SPEAKER_STITCH_THRESHOLD:
                old_centroid = stitch_centroids[assigned_label]
                updated_centroid = torch.nn.functional.normalize(
                    SPEAKER_STITCH_EMA_ALPHA * emb + (1 - SPEAKER_STITCH_EMA_ALPHA) * old_centroid, p=2, dim=0
                )
                stitch_centroids[assigned_label] = updated_centroid
                hist = stitch_histories.setdefault(assigned_label, [])
                hist.append(emb)
                if len(hist) > SPEAKER_STITCH_POOL_SIZE:
                    hist.pop(0)
            else:
                assigned_label = new_canonical_label()
                stitch_centroids[assigned_label] = emb
                stitch_histories[assigned_label] = [emb]

            # Проставляем метку сегменту и запоминаем mapping для этой исходной метки
            seg["speaker"] = assigned_label
            stitched.append(seg)

            if orig_label is not None:
                orig2canon[orig_label] = assigned_label
                orig_emb_count[orig_label] = orig_emb_count.get(orig_label, 0) + 1

        # --- финальный merge похожих канонических центроидов (как раньше) ---
        label_centroids: Dict[str, torch.Tensor] = {}
        for label, hist in stitch_histories.items():
            centroid = torch.stack(hist).mean(dim=0)
            centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
            label_centroids[label] = centroid

        adj: Dict[str, set] = {label: set() for label in label_centroids}
        labels = list(label_centroids.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                sim = torch.dot(label_centroids[a], label_centroids[b]).item()
                if sim >= SPEAKER_STITCH_MERGE_THRESHOLD:
                    adj[a].add(b)
                    adj[b].add(a)

        visited, merge_map = set(), {}
        for label in adj:
            if label in visited:
                continue
            stack, component = [label], []
            while stack:
                l = stack.pop()
                if l in visited:
                    continue
                visited.add(l)
                component.append(l)
                stack.extend(adj[l] - visited)
            if len(component) > 1:
                rep = sorted(component)[0]
                for l in component:
                    merge_map[l] = rep

        if merge_map:
            for seg in stitched:
                old = seg["speaker"]
                if old in merge_map:
                    new = merge_map[old]
                    if new != old:
                        seg["speaker"] = new

        return stitched

    except Exception as e:
        logger.warning(f"[{upload_id}] speaker stitching failed, falling back to original diarization labels: {e}")
        return raw

# + NEW: helper определяет, что этот процесс — "чистый" webhooks-воркер
def _is_webhooks_worker() -> bool:
    # стандартные имена env-переменных, через которые часто пробрасывают список очередей
    q = (
        os.environ.get("CELERY_QUEUES")
        or os.environ.get("CELERY_WORKER_QUEUES")
        or os.environ.get("CELERYD_QUEUES")
    )
    if q:
        qs = [x.strip() for x in q.split(",") if x.strip()]
        # считаем webhooks-воркером, если он слушает ТОЛЬКО webhooks
        if qs and all(x == "webhooks" for x in qs):
            return True

    # запасной путь: явная роль из настроек, если используешь
    role = str(getattr(settings, "WORKER_ROLE", "")).lower()
    if role == "webhooks":
        return True

    # можно также распознать по имени воркера, если в нем есть "webhooks"
    name = (os.environ.get("CELERY_WORKER_NAME") or os.environ.get("HOSTNAME") or "").lower()
    if "webhooks" in name:
        return True

    return False

# ️ было:
# @worker_process_init.connect
# def preload_on_startup(**kwargs):
#     if _HF_AVAILABLE:
#         get_whisper_model()
#     if _PN_AVAILABLE:
#         get_diarization_pipeline()

#  стало:
@worker_process_init.connect
def preload_on_startup(**kwargs):
    # webhooks-воркер не грузит модели
    if _is_webhooks_worker():
        logger.info("[PRELOAD] webhooks worker detected → skip model preload")
        return
    if _HF_AVAILABLE:
        get_whisper_model()
    # ВАЖНО: если используем DiariZen — не трогаем pyannote на этапе прелоада
    if _PN_AVAILABLE and not bool(getattr(settings, "USE_DIARIZEN", True)):
        get_diarization_pipeline()
# ---------------------- Post-processing for diarization ----------------------

def _merge_short_segments(segments: List[Dict[str, Any]],
                          min_dur: float = 0.35,
                          max_gap_to_merge: float = 0.2) -> List[Dict[str, Any]]:
    """
    Сливает слишком короткие сегменты (<min_dur) к ближайшему соседу
    с тем же спикером, иначе — к более длительному соседу.
    Также склеивает соседние сегменты одного спикера, разделённые крошечной паузой (<max_gap_to_merge).
    """
    if not segments:
        return segments

    segs = sorted(segments, key=lambda s: (s["start"], s["end"]))
    # 1) склеить микропауы между одинаковыми спикерами
    merged: List[Dict[str, Any]] = []
    prev = None
    for s in segs:
        if prev and prev["speaker"] == s["speaker"] and (s["start"] - prev["end"]) <= max_gap_to_merge:
            prev["end"] = max(prev["end"], s["end"])
        else:
            if prev:
                merged.append(prev)
            prev = dict(s)
    if prev:
        merged.append(prev)

    # 2) перекинуть очень короткие сегменты
    if len(merged) <= 1:
        return merged

    def dur(x): return x["end"] - x["start"]
    i = 0
    while i < len(merged):
        s = merged[i]
        if dur(s) >= min_dur:
            i += 1
            continue
        # кандидаты — соседи
        left = merged[i - 1] if i - 1 >= 0 else None
        right = merged[i + 1] if i + 1 < len(merged) else None

        # приоритет: сосед с тем же спикером, иначе — более длинный
        def choose_target():
            same_left = left and left["speaker"] == s["speaker"]
            same_right = right and right["speaker"] == s["speaker"]
            if same_left and same_right:
                return left if dur(left) >= dur(right) else right
            if same_left:
                return left
            if same_right:
                return right
            # иначе — длиннейший из соседей
            if left and right:
                return left if dur(left) >= dur(right) else right
            return left or right

        tgt = choose_target()
        if not tgt:
            i += 1
            continue

        # присоединяем по времени
        if tgt is left:
            left["end"] = max(left["end"], s["end"])
            merged.pop(i)
            # не двигаем i, т.к. текущий индекс стал слит с предыдущим
        elif tgt is right:
            right["start"] = min(right["start"], s["start"])
            merged.pop(i)
            # следующий стал шире, остаёмся на i, чтобы переоценить
        else:
            i += 1

    # финальная склейка подряд идущих одинаковых спикеров
    final_out: List[Dict[str, Any]] = []
    prev = None
    for s in merged:
        if prev and prev["speaker"] == s["speaker"] and (s["start"] - prev["end"]) <= max_gap_to_merge:
            prev["end"] = max(prev["end"], s["end"])
        else:
            if prev:
                final_out.append(prev)
            prev = dict(s)
    if prev:
        final_out.append(prev)
    return final_out

# --- Celery tasks ---

@app.task(bind=True, queue="transcribe_cpu")
def convert_to_wav_and_preview(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] convert_to_wav_and_preview received")
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    r.publish(f"progress:{upload_id}", json.dumps({"status": "processing_started"}))
    deliver_webhook.delay("processing_started", upload_id, None)

    try:
        logger.info(f"[{upload_id}] preparing WAV")
        prepare_wav(upload_id)
        logger.info(f"[{upload_id}] WAV ready")
    except Exception as e:
        r.publish(f"progress:{upload_id}", json.dumps({"status": "error", "error": str(e)}))
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    preview_transcribe.delay(upload_id, correlation_id)

@app.task(bind=True, queue="transcribe_gpu")
def preview_transcribe(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] preview_transcribe received")
    try:
        inspector = app.control.inspect()
        # сколько тяжёлых сейчас именно на GPU (без учёта этой задачи)
        heavy_gpu = _count_gpu_heavy_tasks(inspector, exclude_task_id=self.request.id)

        # лимит GPU слотов на «тяжёлые» задачи
        # можно тоньше: отдельно транс/диар, но базово ок суммарный лимит
        limit = int(getattr(settings, "GPU_CONCURRENCY_TRANSCRIBE", 1)) + int(
            getattr(settings, "GPU_CONCURRENCY_DIARIZE", 1))
        limit = max(1, limit)  # защита от 0

        if heavy_gpu >= limit and self.request.delivery_info.get("routing_key") != "transcribe_cpu":
            logger.info(f"[{upload_id}] GPUs busy: heavy_gpu={heavy_gpu} >= limit={limit} → reschedule to CPU")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
        else:
            logger.info(f"[{upload_id}] GPU check ok: heavy_gpu={heavy_gpu} < limit={limit} (stay on GPU)")
    except Exception:
        logger.warning(f"[{upload_id}] failed to inspect workers for fallback logic")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    proc = prepare_preview_segment(upload_id)
    model = get_whisper_model()
    segments_gen, _ = model.transcribe(
        proc.stdout,
        word_timestamps=True,  # для превью оставляем пометки по словам
        beam_size=1,
        best_of=1,
        temperature=0.0,
        **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
    )
    proc.stdout.close(); proc.wait()
    segments = list(segments_gen)
    for seg in segments:
        r.publish(
            f"progress:{upload_id}",
            json.dumps({
                "status": "preview_partial",
                "fragment": {"start": seg.start, "end": seg.end, "text": seg.text}
            })
        )
    preview = {
        "text": "".join(s.text for s in segments),
        "timestamps": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "preview_transcript.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    r.publish(f"progress:{upload_id}", json.dumps({"status": "preview_done", "preview": preview}))
    deliver_webhook.delay("preview_completed", upload_id, {"preview": preview})
    transcribe_segments.delay(upload_id, correlation_id)


@app.task(bind=True, queue="transcribe_gpu")
def transcribe_segments(self, upload_id, correlation_id):
    logger.info(f"[{upload_id}] transcribe_segments received")
    try:
        import torch
        logger.info(f"[{upload_id}] GPU memory reserved before transcription: "
                    f"{torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

    # Если GPU перегружены — уходим на CPU
    try:
        inspector = app.control.inspect()
        # сколько тяжёлых сейчас именно на GPU (без учёта этой задачи)
        heavy_gpu = _count_gpu_heavy_tasks(inspector, exclude_task_id=self.request.id)

        # лимит GPU слотов на «тяжёлые» задачи
        # можно тоньше: отдельно транс/диар, но базово ок суммарный лимит
        limit = int(getattr(settings, "GPU_CONCURRENCY_TRANSCRIBE", 1)) + int(
            getattr(settings, "GPU_CONCURRENCY_DIARIZE", 1))
        limit = max(1, limit)  # защита от 0

        if heavy_gpu >= limit and self.request.delivery_info.get("routing_key") != "transcribe_cpu":
            logger.info(f"[{upload_id}] GPUs busy: heavy_gpu={heavy_gpu} >= limit={limit} → reschedule to CPU")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
        else:
            logger.info(f"[{upload_id}] GPU check ok: heavy_gpu={heavy_gpu} < limit={limit} (stay on GPU)")
    except Exception:
        logger.warning(f"[{upload_id}] failed to inspect workers for fallback logic")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    wav, duration = prepare_wav(upload_id)
    if not _HF_AVAILABLE:
        logger.error(f"[{upload_id}] whisper model unavailable, failing")
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    model = get_whisper_model()
    raw_segs: List[Any] = []

    def _transcribe_with_vad(source, offset: float = 0.0, core_start: Optional[float] = None, core_end: Optional[float] = None):
        segs, _ = model.transcribe(
            source,
            word_timestamps=WHISPER_WORD_TIMESTAMPS,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": int(getattr(settings, "SENTENCE_MAX_GAP_S", 0.35) * 1000),
                "speech_pad_ms": 200,
            },
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREV_TEXT,
            **({"language": settings.WHISPER_LANGUAGE} if settings.WHISPER_LANGUAGE else {}),
        )
        result = []
        for s in segs:
            s.start += offset
            s.end += offset
            # если задано ядро чанка — отрежем только попадающее внутрь ядра
            if core_start is not None and core_end is not None:
                if s.end <= core_start or s.start >= core_end:
                    continue
                s.start = max(s.start, core_start)
                s.end = min(s.end, core_end)
                if s.end <= s.start:
                    continue
            result.append(s)
        return result

    # короткое аудио — одним проходом
    if duration <= settings.VAD_MAX_LENGTH_S:
        logger.info(f"[{upload_id}] short audio ({duration:.1f}s) → single VAD pass")
        raw_segs = _transcribe_with_vad(str(wav))
    else:
        total_chunks = math.ceil(duration / settings.CHUNK_LENGTH_S)
        processed_key = f"transcribe:processed_chunks:{upload_id}"
        processed = {int(x) for x in r.smembers(processed_key)}

        pad = float(getattr(settings, "TRANSCRIPTION_CHUNK_PAD_S", TRANSCRIPTION_CHUNK_PAD_S))
        chunk_idx = 0
        core_start = 0.0

        while core_start < duration:
            if chunk_idx in processed:
                logger.info(f"[{upload_id}] skip chunk {chunk_idx+1}/{total_chunks} (already done)")
                core_start += settings.CHUNK_LENGTH_S
                chunk_idx += 1
                continue

            core_len = min(settings.CHUNK_LENGTH_S, duration - core_start)
            core_end = core_start + core_len

            left_pad = pad if core_start > 0 else 0.0
            right_pad = pad if core_end < duration else 0.0

            read_start = max(0.0, core_start - left_pad)
            read_end = min(duration, core_end + right_pad)
            read_len = max(0.0, read_end - read_start)

            logger.info(
                f"[{upload_id}] transcribe chunk {chunk_idx+1}/{total_chunks}: "
                f"{read_start:.1f}s→{read_end:.1f}s (core {core_start:.1f}s→{core_end:.1f}s, "
                f"pad L={left_pad:.1f}s R={right_pad:.1f}s)"
            )

            try:
                p = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-threads", str(settings.FFMPEG_THREADS),
                        "-ss", str(read_start), "-t", str(read_len),
                        "-i", str(wav),
                        "-f", "wav", "pipe:1"
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                chunk_segs = _transcribe_with_vad(
                    p.stdout, offset=read_start, core_start=core_start, core_end=core_end
                )
                p.stdout.close(); p.wait()

                raw_segs.extend(chunk_segs)
                r.sadd(processed_key, chunk_idx)
            except Exception as e:
                logger.error(f"[{upload_id}] error in transcribe chunk {chunk_idx+1}/{total_chunks}: {e}", exc_info=True)
                try:
                    import torch; torch.cuda.empty_cache()
                except ImportError:
                    pass
            finally:
                core_start += core_len
                chunk_idx += 1

        r.delete(processed_key)

    # группируем в предложения
    flat = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segs]
    flat.sort(key=lambda x: x["start"])
    sentences = group_into_sentences(flat)

    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
    logger.info(f"[{upload_id}] transcription completed ({len(sentences)} sentences)")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "transcript_done"}))
    deliver_webhook.delay("transcription_completed", upload_id, {"transcript": sentences})

    try:
        import torch
        logger.info(f"[{upload_id}] GPU memory reserved after transcription: "
                    f"{torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass


@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    """
    Диаризация с приоритетом DiariZen:
      1) BUT-FIT/diarizen-wavlm-large-s80-mlc (целиком).
      2) Фолбэк на pyannote/speaker-diarization-3.1 (с чанкингом/паддингом).
      3) Последний шанс: один сегмент 0..duration (один спикер), чтобы не падать.
    Постобработка: дроп коротких, склейка пауз, анти-flip, схлопывание подряд идущих.
    """
    import gc, json, math, subprocess, torch
    from pathlib import Path
    from typing import Optional, List, Dict, Any
    from redis import Redis

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full started")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    deliver_webhook.delay("diarization_started", upload_id, None)

    try:
        logger.info(
            f"[{upload_id}] GPU memory reserved before diarization: "
            f"{torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}"
        )
    except Exception:
        pass

    # --- Подготовка WAV и длительность
    wav, duration = prepare_wav(upload_id)

    # --- Настройки чанкинга/паддинга (для pyannote-ветки)
    try:
        chunk_limit = int(getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 300) or 0)
    except Exception:
        logger.warning(f"[{upload_id}] invalid DIARIZATION_CHUNK_LENGTH_S, fallback to 0")
        chunk_limit = 0
    pad = float(getattr(settings, "DIARIZATION_CHUNK_PADDING_S", 10.0) or 0.0)
    use_chunking = bool(chunk_limit and duration > chunk_limit)
    total_chunks = math.ceil(duration / chunk_limit) if use_chunking else 1

    # --- Подсказки для pyannote
    infer_kwargs: Dict[str, Any] = {}
    try:
        if getattr(settings, "PYANNOTE_NUM_SPEAKERS", None):
            infer_kwargs["num_speakers"] = int(settings.PYANNOTE_NUM_SPEAKERS)
        else:
            v = getattr(settings, "PYANNOTE_MIN_SPEAKERS", None)
            if v is not None:
                infer_kwargs["min_speakers"] = int(v)
            v = getattr(settings, "PYANNOTE_MAX_SPEAKERS", None)
            if v is not None:
                infer_kwargs["max_speakers"] = int(v)
    except Exception:
        pass

    used_backend = None
    diar_segments: List[Dict[str, Any]] = []

    # --------- helpers
    def _patch_diarizen_config(pipeline) -> None:
        """
        Безопасно патчит pipeline.config/clustering.args для auto speakers.
        Поддерживает dict и OmegaConf/объектный доступ. Ничего не бросает наружу.
        """
        try:
            # вычисляем целевые значения
            fixed = getattr(settings, "DIARIZEN_NUM_SPEAKERS", None)
            if fixed is not None and int(fixed) > 0:
                min_spk = max_spk = int(fixed)
            else:
                min_spk = int(getattr(settings, "DIARIZEN_MIN_SPEAKERS", 1) or 1)
                max_spk = int(getattr(settings, "DIARIZEN_MAX_SPEAKERS", 8) or 8)
                if max_spk < min_spk:
                    max_spk = min_spk

            thr = None
            try:
                # поддерживаем старое имя переменной: DIARIZEN_CLUSTER_THRESHOLD
                thr = float(getattr(settings, "DIARIZEN_CLUSTER_THRESHOLD", None))
            except Exception:
                thr = None

            mcs = getattr(settings, "DIARIZEN_MIN_CLUSTER_SIZE", None)
            mcs = int(mcs) if mcs is not None else None

            cfg = getattr(pipeline, "config", None)

            def _apply_to_args(_args: Any):
                try:
                    # dict-путь
                    if isinstance(_args, dict):
                        _args["min_speakers"] = min_spk
                        _args["max_speakers"] = max_spk
                        if thr is not None:
                            _args["ahc_threshold"] = thr
                            _args["cluster_threshold"] = thr
                        if mcs is not None:
                            _args["min_cluster_size"] = mcs
                        return True
                    # объектный путь (OmegaConf / DictConfig)
                    if _args is not None:
                        setattr(_args, "min_speakers", min_spk)
                        setattr(_args, "max_speakers", max_spk)
                        if thr is not None:
                            setattr(_args, "ahc_threshold", thr)
                            setattr(_args, "cluster_threshold", thr)
                        if mcs is not None:
                            setattr(_args, "min_cluster_size", mcs)
                        return True
                except Exception:
                    return False
                return False

            changed = False
            # 1) config.clustering.args
            if isinstance(cfg, dict):
                cl = cfg.setdefault("clustering", {})
                args = cl.setdefault("args", {})
                if _apply_to_args(args):
                    changed = True
            else:
                try:
                    cl = getattr(cfg, "clustering", None)
                    args = getattr(cl, "args", None) if cl is not None else None
                    if _apply_to_args(args):
                        changed = True
                except Exception:
                    pass

            # 2) runtime pipeline.clustering.args (если есть)
            try:
                clrt = getattr(pipeline, "clustering", None)
                clrt_args = getattr(clrt, "args", None)
                if _apply_to_args(clrt_args):
                    changed = True
            except Exception:
                pass

            logger.info(
                f"[DIARIZE] DiariZen speaker range: min={min_spk}, max={max_spk}"
                + (f", thr={thr}" if thr is not None else "")
                + (", config_patched" if changed else ", config_unchanged")
            )
        except Exception as e:
            logger.warning(f"[{upload_id}] could not adjust DiariZen clustering params: {e}")

    def _run_diarizen(_wav: Path) -> Optional[List[Dict[str, Any]]]:
        """Одна проходка DiariZen на весь файл."""
        try:
            from diarizen.pipelines.inference import DiariZenPipeline  # type: ignore
        except Exception as e:
            logger.warning(f"[{upload_id}] DiariZen import failed: {e}")
            return None

        model_id = getattr(settings, "DIARIZEN_MODEL_ID", "BUT-FIT/diarizen-wavlm-large-s80-mlc")
        try:
            pipeline = DiariZenPipeline.from_pretrained(model_id)
            # патчим конфиг (auto speakers)
            _patch_diarizen_config(pipeline)
            # опционально выставим устройство
            try:
                dev = getattr(settings, "DIARIZER_DEVICE", None) or ("cuda" if torch.cuda.is_available() else "cpu")
                if hasattr(pipeline, "to"):
                    pipeline.to(torch.device(dev))  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"[{upload_id}] DiariZen .to(device) failed (will run as-is): {e}")
        except Exception as e:
            logger.warning(f"[{upload_id}] DiariZen load failed for {model_id}: {e}")
            return None

        try:
            with torch.inference_mode():
                # Важно: __call__ НЕ принимает min/max_speakers — никаких kwargs!
                ann = pipeline(str(_wav))
        except Exception as e:
            logger.error(f"[{upload_id}] DiariZen inference failed: {e}", exc_info=True)
            return None

        out: List[Dict[str, Any]] = []
        try:
            for s, _, spk in ann.itertracks(yield_label=True):
                out.append({"start": float(s.start), "end": float(s.end), "speaker": spk})
            out.sort(key=lambda x: x["start"])
            return out
        except Exception as e:
            logger.error(f"[{upload_id}] DiariZen output parse failed: {e}", exc_info=True)
            return None
        finally:
            try:
                del ann
            except Exception:
                pass

    def _run_pyannote(_wav: Path) -> Optional[List[Dict[str, Any]]]:
        """pyannote с чанками и паддингом. Возвращает None, если пайплайн недоступен."""
        try:
            from pyannote.audio import Pipeline as PyannotePipeline  # noqa
        except Exception as e:
            logger.warning(f"[{upload_id}] pyannote import failed: {e}")
            return None

        # пробуем собрать пайплайн (в окружениях с vendored pyannote это может падать)
        try:
            pipeline = get_diarization_pipeline()  # у тебя уже есть эта функция
        except Exception as e:
            logger.error(f"[{upload_id}] pyannote pipeline unavailable: {e}")
            return None

        res: List[Dict[str, Any]] = []
        if use_chunking:
            processed_key = f"diarize:processed_chunks:{upload_id}"
            processed = {int(x) for x in r.smembers(processed_key)}

            offset = 0.0
            chunk_idx = 0
            while offset < duration:
                core_len = min(chunk_limit, duration - offset)
                left_pad = pad if offset > 0 else 0.0
                right_pad = pad if (offset + core_len) < duration else 0.0
                st = max(0.0, offset - left_pad)
                en = min(duration, offset + core_len + right_pad)

                if chunk_idx in processed:
                    logger.info(f"[{upload_id}] skip diarize chunk {chunk_idx+1}/{total_chunks}")
                    offset += core_len
                    chunk_idx += 1
                    continue

                logger.info(
                    f"[{upload_id}] diarize chunk {chunk_idx+1}/{total_chunks}: "
                    f"{st:.1f}s→{en:.1f}s (core {offset:.1f}s→{offset+core_len:.1f}s, pad L={left_pad:.1f}s R={right_pad:.1f}s)"
                )

                tmp = Path(getattr(settings, "DIARIZER_CACHE_DIR", "/tmp")) / f"{upload_id}_chunk_{chunk_idx}.wav"
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-threads", str(max(1, int(getattr(settings, "FFMPEG_THREADS", 2)) // 2)),
                        "-ss", str(st), "-t", str(en - st),
                        "-i", str(_wav), str(tmp),
                    ],
                    check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
                )

                try:
                    with torch.inference_mode():
                        ann = pipeline(str(tmp), **infer_kwargs)

                    before = len(res)
                    for s, _, spk in ann.itertracks(yield_label=True):
                        g0, g1 = float(s.start) + st, float(s.end) + st
                        # вырезаем только «ядро» чанка, чтобы не дублировать overlap
                        if g1 <= offset or g0 >= (offset + core_len):
                            continue
                        g0 = max(g0, offset)
                        g1 = min(g1, offset + core_len)
                        if g1 > g0:
                            res.append({"start": g0, "end": g1, "speaker": spk})

                    added = len(res) - before
                    logger.info(f"[{upload_id}] diarize chunk {chunk_idx+1}/{total_chunks} done: added {added} segments")
                    r.sadd(processed_key, chunk_idx)
                except Exception as e:
                    logger.error(f"[{upload_id}] error in diarize chunk {chunk_idx+1}/{total_chunks}: {e}", exc_info=True)
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        del ann  # type: ignore
                    except Exception:
                        pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                offset += core_len
                chunk_idx += 1

            r.delete(processed_key)
        else:
            logger.info(f"[{upload_id}] Short audio or chunking disabled, single diarization pass")
            with torch.inference_mode():
                ann = pipeline(str(_wav), **infer_kwargs)
            for s, _, spk in ann.itertracks(yield_label=True):
                res.append({"start": float(s.start), "end": float(s.end), "speaker": spk})
            try:
                del ann
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        res.sort(key=lambda x: x["start"])
        return res

    def _degenerate_one_speaker() -> List[Dict[str, Any]]:
        """Последний шанс: не падаем, а возвращаем один сегмент на весь файл."""
        speaker_label = getattr(settings, "DIARIZATION_DEGRADED_SPEAKER", "SPEAKER_00")
        logger.warning(f"[{upload_id}] DEGRADATION: returning single full-span segment (no diarization backend available)")
        return [{"start": 0.0, "end": float(duration), "speaker": speaker_label}]

    # --------- Запуск: 1) DiariZen
    try:
        if bool(getattr(settings, "USE_DIARIZEN", True)):
            attempt = _run_diarizen(wav)
            if attempt:
                diar_segments = attempt
                used_backend = "diarizen"
            else:
                logger.warning(f"[{upload_id}] DiariZen unavailable → fallback to pyannote")
    except Exception as e:
        logger.warning(f"[{upload_id}] DiariZen unexpected error (fallback to pyannote): {e}")

    # --------- 2) pyannote
    if not diar_segments:
        try:
            attempt = _run_pyannote(wav)
            if attempt:
                diar_segments = attempt
                used_backend = "pyannote"
        except Exception as e:
            logger.error(f"[{upload_id}] diarization failed in fallback: {e}", exc_info=True)

    # --------- 3) Деградация (гарантируем отсутствие падений)
    if not diar_segments:
        if bool(getattr(settings, "DIARIZATION_ALLOW_DEGRADED", True)):
            diar_segments = _degenerate_one_speaker()
            used_backend = "degraded"
        else:
            logger.error(f"[{upload_id}] empty diarization result and degraded mode disabled")
            deliver_webhook.delay("processing_failed", upload_id, None)
            return

    # ------- Постобработка
    MIN_SEG = float(getattr(settings, "DIARIZATION_MIN_SEGMENT_S", 0.20))
    GAP_MERGE = float(getattr(settings, "DIARIZATION_MERGE_GAP_S", 0.20))
    ISLAND_MAX = float(getattr(settings, "DIARIZATION_ISLAND_MAX_S", 0.60))

    diar_segments = [s for s in diar_segments if (s["end"] - s["start"]) >= MIN_SEG]

    merged: List[Dict[str, Any]] = []
    cur = None
    for seg in sorted(diar_segments, key=lambda x: x["start"]):
        if cur and cur["speaker"] == seg["speaker"] and (seg["start"] - cur["end"]) <= GAP_MERGE:
            cur["end"] = max(cur["end"], seg["end"])
        else:
            if cur:
                merged.append(cur)
            cur = dict(seg)
    if cur:
        merged.append(cur)
    diar_segments = merged

    def _stabilize_labels(segments: List[Dict[str, Any]], island_max: float = ISLAND_MAX) -> List[Dict[str, Any]]:
        if len(segments) < 3:
            return segments
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(segments):
            if 0 < i < len(segments) - 1:
                prev, cur, nxt = segments[i - 1], segments[i], segments[i + 1]
                cur_dur = cur["end"] - cur["start"]
                if prev["speaker"] == nxt["speaker"] != cur["speaker"] and cur_dur <= island_max:
                    # поглощаем островок
                    if out:
                        out.pop()
                    out.append({"start": prev["start"], "end": nxt["end"], "speaker": prev["speaker"]})
                    i += 2
                    continue
            out.append(segments[i])
            i += 1
        return out

    diar_segments = _stabilize_labels(diar_segments)

    final_out: List[Dict[str, Any]] = []
    buf: Optional[Dict[str, Any]] = None
    for seg in sorted(diar_segments, key=lambda x: x["start"]):
        if buf and buf["speaker"] == seg["speaker"] and (seg["start"] - buf["end"]) < 0.10:
            buf["end"] = seg["end"]
        else:
            if buf:
                final_out.append(buf)
            buf = dict(seg)
    if buf:
        final_out.append(buf)

    # ------- Сохранение/ивенты
    out_dir = Path(settings.RESULTS_FOLDER) / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diarization.json").write_text(json.dumps(final_out, ensure_ascii=False, indent=2))

    speakers = sorted({seg["speaker"] for seg in final_out if seg.get("speaker") is not None})
    logger.info(
        f"[{upload_id}] diarization_done by {used_backend}, "
        f"total segments: {len(final_out)}, speakers_detected={len(speakers)}"
    )

    r.publish(
        f"progress:{upload_id}",
        json.dumps({
            "status": "diarization_done",
            "backend": used_backend,
            "segments": len(final_out),
            "speakers": len(speakers),
            "total_chunks": total_chunks
        })
    )
    deliver_webhook.delay("diarization_completed", upload_id, {"diarization": final_out})

    try:
        logger.info(f"[{upload_id}] GPU memory reserved after diarization: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except Exception:
        pass


@app.task(
    bind=True,
    name="deliver_webhook",
    queue="webhooks",
    max_retries=5,
    default_retry_delay=30,
)
def deliver_webhook(self, event_type: str, upload_id: str, data: Optional[Any]):
    url = settings.WEBHOOK_URL
    secret = settings.WEBHOOK_SECRET
    if not url or not secret:
        return

    payload = {
        "event_type": event_type,
        "upload_id": upload_id,
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "data": data,
    }
    headers = {
        "Content-Type": "application/json",
        "X-WebHook-Secret": secret,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=(5, 30))
        code = resp.status_code
        if 200 <= code < 300 or code == 405:
            logger.info(f"[WEBHOOK] {event_type} succeeded for {upload_id} ({code})")
            return
        if 400 <= code < 500:
            logger.error(f"[WEBHOOK] {event_type} returned {code} for {upload_id}, aborting")
            return
        raise Exception(f"Webhook returned {code}")
    except Exception as exc:
        logger.warning(f"[WEBHOOK] {event_type} error for {upload_id}, retrying: {exc}")
        raise self.retry(exc=exc)


@app.task(bind=True, queue="transcribe_cpu")
def cleanup_old_files(self):
    age = settings.FILE_RETENTION_DAYS
    deleted = 0
    for base in (Path(settings.UPLOAD_FOLDER), Path(settings.RESULTS_FOLDER)):
        for p in base.glob("**/*"):
            try:
                if datetime.utcnow() - datetime.fromtimestamp(p.stat().st_mtime) > timedelta(days=age):
                    if p.is_dir():
                        p.rmdir()
                    else:
                        p.unlink()
                    deleted += 1
            except Exception:
                continue
    logger.info(f"[CLEANUP] deleted {deleted} old files")