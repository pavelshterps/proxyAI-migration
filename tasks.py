import json
import logging
import subprocess
import time
import re
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

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
_speaker_embedding_model = None  # type: ignore

USE_VAD_IN_FULL = bool(getattr(settings, "USE_VAD_IN_FULL", False))
TRANSCRIBE_OVERLAP_S = float(getattr(settings, "TRANSCRIBE_OVERLAP_S", 0.5))

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

# --- audio cleanup parameters (can be overridden in env via settings if добавишь позже) ---
AUDIO_HP_F = int(getattr(settings, "AUDIO_HP_F", 120))  # Hz
AUDIO_AFFTDN_NR = int(getattr(settings, "AUDIO_AFFTDN_NR", 12))  # 6..24 мягко
AUDIO_USE_LOUDNORM = bool(getattr(settings, "AUDIO_USE_LOUDNORM", True))
AUDIO_LOUDNORM = getattr(settings, "AUDIO_LOUDNORM", "I=-23:TP=-2:LRA=11")  # broadcast-ish

def _ffmpeg_filter_chain() -> str:
    chain = [f"highpass=f={AUDIO_HP_F}", f"afftdn=nr={AUDIO_AFFTDN_NR}"]
    if AUDIO_USE_LOUDNORM:
        chain.append(f"loudnorm={AUDIO_LOUDNORM}")
    return ",".join(chain)

def prepare_wav(upload_id: str) -> (Path, float):
    """
    Готовит / нормализует WAV 16kHz mono PCM.
    Безопасно обрабатывает случай, когда вход уже .wav (не пишет "поверх").
    Пробует мягкий денойз+highpass+loudnorm, при неудаче — plain ресемпл.
    """
    # исходник (любое расширение)
    src = next(Path(settings.UPLOAD_FOLDER).glob(f"{upload_id}.*"))
    target = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.wav"
    tmp_out = Path(settings.UPLOAD_FOLDER) / f"{upload_id}.tmp.wav"

    info = probe_audio(src)
    duration = float(info.get("duration", 0.0) or 0.0)

    # если уже готовый wav 16k mono pcm — просто убедимся в названии
    if (
        src.suffix.lower() == ".wav"
        and info.get("codec_name") == "pcm_s16le"
        and int(info.get("sample_rate", 0)) == 16000
        and int(info.get("channels", 0)) == 1
    ):
        if src != target:
            # переименуем в целевое имя
            src.rename(target)
        return target, duration

    # Готовим входной путь для ffmpeg
    in_path = str(src)

    # Всегда пишем во временный файл (никогда не пишем поверх входа)
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass

    # Попытка 1: фильтры (мягкие)
    # highpass режет гул/низ, afftdn — частотный денойз,
    # loudnorm — нормализация громкости (однопроходный режим).
    filters = "highpass=f=120,afftdn=nr=12,loudnorm=I=-23:TP=-2:LRA=11"
    ffmpeg_base = [
        "ffmpeg", "-y",
        "-threads", str(settings.FFMPEG_THREADS),
        "-hide_banner", "-nostdin",
        "-i", in_path,
        "-vn",  # только аудио
    ]

    try:
        cmd = ffmpeg_base + [
            "-af", filters,
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            str(tmp_out),
        ]
        res = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(f"[{upload_id}] ffmpeg filtered OK")
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"[{upload_id}] ffmpeg filters failed ({e}). "
            f"stderr: {e.stderr[:5000] if e.stderr else 'no-stderr'}. "
            f"Falling back to plain resample."
        )
        # Попытка 2: чистая перекодировка без фильтров
        try:
            cmd2 = ffmpeg_base + [
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                str(tmp_out),
            ]
            res2 = subprocess.run(
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
                f"stderr: {e2.stderr[:5000] if e2.stderr else 'no-stderr'}"
            )
            raise

    # Атомарно заменим целевой файл
    try:
        if target.exists():
            target.unlink()
    except Exception:
        pass
    tmp_out.replace(target)

    # Обновим длительность после перекодировки (по желанию можно не делать)
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
    SILENCE_GAP_S = getattr(settings, "SENTENCE_MAX_GAP_S", 0.5)
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
    global _diarization_pipeline
    if _diarization_pipeline is None:
        model_id = getattr(settings, "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1")
        pipe = PyannotePipeline.from_pretrained(
            model_id,
            use_auth_token=settings.HUGGINGFACE_TOKEN,
            cache_dir=settings.DIARIZER_CACHE_DIR,
        )
        # <- ВАЖНО: на GPU
        try:
            import torch
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
        except Exception:
            pass
        _diarization_pipeline = pipe
        logger.info(f"[DIARIZE] loaded pipeline {model_id}")
    return _diarization_pipeline

def get_speaker_embedding_model():
    global _speaker_embedding_model
    if _speaker_embedding_model is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            import torch  # NEW
        except ImportError as e:
            logger.warning(f"[STITCH] speechbrain not available, cannot do speaker stitching: {e}")
            raise

        savedir = Path(settings.DIARIZER_CACHE_DIR) / "spkrec-ecapa-voxceleb"

        # NEW: если доступен CUDA — явно просим SpeechBrain работать на GPU
        run_opts = {}
        try:
            if torch.cuda.is_available():
                run_opts = {"device": "cuda"}
        except Exception:
            pass

        _speaker_embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts=run_opts  # NEW
        )
        logger.info("[STITCH] loaded ECAPA (SpeechBrain) with run_opts=%s", run_opts or "{}")
    return _speaker_embedding_model

def stitch_speakers(raw: List[Dict[str, Any]], wav: Path, upload_id: str) -> List[Dict[str, Any]]:
    if not SPEAKER_STITCH_ENABLED:
        return raw

    unique_orig = set(seg.get("speaker") for seg in raw)
    if len(unique_orig) <= 1:
        logger.debug(f"[{upload_id}] only one original speaker {unique_orig}, skipping stitching")
        return raw

    try:
        import torch
        import torchaudio
        import torch.nn.functional as F
        from torch.utils.data import DataLoader  # NEW

        model = get_speaker_embedding_model()

        waveform, sr = torchaudio.load(str(wav))
        if sr != 16000:
            from torchaudio.transforms import Resample
            resampler = Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # NEW: параметры извлечения эмбеддингов
        WIN_SEC = float(getattr(settings, "STITCH_EMB_WINDOW_S", 1.5))       # центр-окно, даёт устойчивость
        MAX_SEC = float(getattr(settings, "STITCH_EMB_MAX_WINDOW_S", 3.0))   # ограничим длину окна
        HOP_STRIDE = float(getattr(settings, "STITCH_EMB_HOP_S", 1.5))       # если сегмент длинный — возьмём несколько окон
        BATCH = int(getattr(settings, "STITCH_EMB_BATCH", 16))               # батчим ради скорости

        def _center_crop(start_s: float, end_s: float) -> List[Tuple[int, int]]:
            """Вернём список [начало, конец) сэмплов под окна для эмбеддинга."""
            dur = end_s - start_s
            if dur <= 0:
                return []
            # если сегмент слишком длинный — берём несколько окон через шаг
            win = min(MAX_SEC, WIN_SEC)
            if dur <= win:
                mid = (start_s + end_s) * 0.5
                s0 = max(0, int((mid - win * 0.5) * sr))
                s1 = min(waveform.size(1), s0 + int(win * sr))
                return [(s0, s1)]
            else:
                # разреженно по центру сегмента
                out = []
                cur = start_s + (dur - win) * 0.5  # старт из центра
                # сдвинем влево/вправо на несколько окон
                starts = [cur + k * HOP_STRIDE for k in (-1, 0, 1)]
                for st in starts:
                    st = max(start_s, min(st, end_s - win))
                    s0 = int(st * sr)
                    s1 = min(waveform.size(1), s0 + int(win * sr))
                    out.append((s0, s1))
                # уникализируем
                uniq = []
                seen = set()
                for s0, s1 in out:
                    key = (s0, s1)
                    if key not in seen:
                        uniq.append(key); seen.add(key)
                return uniq

        # NEW: готовим «задания» на эмбеддинги
        jobs = []
        for seg in sorted(raw, key=lambda x: x["start"]):
            start, end = seg["start"], seg["end"]
            if end <= start:
                continue
            for s0, s1 in _center_crop(start, end):
                if s1 > s0:
                    jobs.append((seg, s0, s1))

        # батч-энкодинг
        embeddings_by_seg: Dict[int, List[torch.Tensor]] = {}
        for i in range(0, len(jobs), BATCH):
            batch = jobs[i:i+BATCH]
            batch_wavs = []
            for _, s0, s1 in batch:
                piece = waveform[:, s0:s1]
                if piece.numel() == 0:
                    # заглушка — пропустим потом
                    batch_wavs.append(torch.zeros(1, int(WIN_SEC * sr)))
                else:
                    batch_wavs.append(piece)

            # паддинг до одной длины — требуется многим энкодерам
            max_len = max(w.size(1) for w in batch_wavs) if batch_wavs else 0
            padded = []
            for w in batch_wavs:
                if w.size(1) < max_len:
                    pad = torch.zeros(1, max_len - w.size(1), dtype=w.dtype, device=w.device)
                    w = torch.cat([w, pad], dim=1)
                padded.append(w)
            if not padded:
                continue
            x = torch.stack(padded, dim=0)  # [B, 1, T]

            with torch.no_grad():
                embs = model.encode_batch(x)  # SpeechBrain вернёт [B, 1, D] или [B, D]
            embs = embs.squeeze(1) if embs.ndim == 3 else embs
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)

            for (seg, _, _), e in zip(batch, embs):
                idx = id(seg)
                embeddings_by_seg.setdefault(idx, []).append(e.cpu())

        # агрегируем по сегменту (среднее)
        seg_centroids: Dict[int, torch.Tensor] = {}
        for seg in raw:
            idx = id(seg)
            lst = embeddings_by_seg.get(idx, None)
            if not lst:
                continue
            c = torch.stack(lst, dim=0).mean(dim=0)
            c = torch.nn.functional.normalize(c, p=2, dim=0)
            seg_centroids[idx] = c

        # онлайн-кластеризация сегментов по косинусной близости (как у тебя, но на усреднённых центрах)
        stitch_centroids: Dict[str, torch.Tensor] = {}
        stitch_histories: Dict[str, List[torch.Tensor]] = {}
        next_label_idx = 0

        def new_canonical_label():
            nonlocal next_label_idx
            label = f"spk_{next_label_idx}"
            next_label_idx += 1
            return label

        stitched: List[Dict[str, Any]] = []
        raw_sorted = sorted(raw, key=lambda x: x["start"])

        for seg in raw_sorted:
            idx = id(seg)
            emb = seg_centroids.get(idx, None)
            if emb is None:
                stitched.append(seg)
                continue

            assigned_label = None
            best_sim = -1.0
            for canon_label, centroid in stitch_centroids.items():
                sim = float(torch.dot(emb, centroid).item())
                if sim > best_sim:
                    best_sim = sim
                    assigned_label = canon_label

            if assigned_label is not None and best_sim >= SPEAKER_STITCH_THRESHOLD:
                old_centroid = stitch_centroids[assigned_label]
                updated_centroid = F.normalize(
                    SPEAKER_STITCH_EMA_ALPHA * emb + (1 - SPEAKER_STITCH_EMA_ALPHA) * old_centroid, p=2, dim=0
                )
                stitch_centroids[assigned_label] = updated_centroid
                hist = stitch_histories[assigned_label]
                hist.append(emb)
                if len(hist) > SPEAKER_STITCH_POOL_SIZE:
                    hist.pop(0)
                logger.debug(f"[{upload_id}] reused speaker {assigned_label} (sim={best_sim:.3f})")
            else:
                assigned_label = new_canonical_label()
                stitch_centroids[assigned_label] = emb
                stitch_histories[assigned_label] = [emb]
                logger.debug(f"[{upload_id}] created new speaker label {assigned_label}")

            seg["speaker"] = assigned_label
            stitched.append(seg)

        # пост-слияние очень похожих центроид
        label_centroids: Dict[str, torch.Tensor] = {}
        for label, hist in stitch_histories.items():
            centroid = torch.stack(hist).mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0)
            label_centroids[label] = centroid

        adj: Dict[str, set] = {label: set() for label in label_centroids}
        labels = list(label_centroids.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                sim = float(torch.dot(label_centroids[a], label_centroids[b]).item())
                if sim >= SPEAKER_STITCH_MERGE_THRESHOLD:
                    adj[a].add(b)
                    adj[b].add(a)

        visited, merge_map = set(), {}
        for label in adj:
            if label in visited: continue
            stack, component = [label], []
            while stack:
                l = stack.pop()
                if l in visited: continue
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
                        logger.debug(f"[{upload_id}] merged speaker {old} -> {new} based on centroid similarity")
                        seg["speaker"] = new

        return stitched
    except Exception as e:
        logger.warning(f"[{upload_id}] speaker stitching failed, falling back to original diarization labels: {e}")
        return raw

@worker_process_init.connect
def preload_on_startup(**kwargs):
    if _HF_AVAILABLE:
        get_whisper_model()
    if _PN_AVAILABLE:
        get_diarization_pipeline()

# ---------------------- Post-processing for diarization ----------------------
# NEW: если короткий сегмент B зажат между A и A — перекрашиваем в A
def _fix_sandwich_flips(segments: List[Dict[str, Any]],
                        max_len: float = 1.0) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    segs = sorted(segments, key=lambda s: s["start"])
    out = []
    for i, s in enumerate(segs):
        if 0 < i < len(segs)-1:
            prev_s, next_s = segs[i-1], segs[i+1]
            if (s["end"] - s["start"]) <= max_len \
               and prev_s["speaker"] == next_s["speaker"] != s["speaker"]:
                # перекрашиваем
                s = dict(s); s["speaker"] = prev_s["speaker"]
        out.append(s)
    return out

# NEW: если короткий «чужой» фрагмент примыкает к длинному «своему» — перекрашиваем
def _glue_stray_shorts(segments: List[Dict[str, Any]],
                       max_len: float = 0.6, max_gap: float = 0.15) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    segs = sorted(segments, key=lambda s: s["start"])
    out = []
    for i, s in enumerate(segs):
        dur = s["end"] - s["start"]
        if dur <= max_len:
            left = segs[i-1] if i-1 >= 0 else None
            right = segs[i+1] if i+1 < len(segs) else None
            def gap(a,b):
                return max(0.0, (b["start"] - a["end"])) if a and b else 1e9
            # если коротыш прижат к одному из соседей и сосед длинный — красим в соседа
            if left and left["speaker"] != s["speaker"] and gap(left,s) <= max_gap and (left["end"]-left["start"]) >= (dur*2):
                s = dict(s); s["speaker"] = left["speaker"]
            elif right and right["speaker"] != s["speaker"] and gap(s,right) <= max_gap and (right["end"]-right["start"]) >= (dur*2):
                s = dict(s); s["speaker"] = right["speaker"]
        out.append(s)
    return out

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
        active = inspector.active() or {}
        heavy = 0
        for node_tasks in active.values():
            for t in node_tasks:
                if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments"):
                    heavy += 1
        if heavy >= 2:
            logger.info(f"[{upload_id}] both GPUs busy (found {heavy} heavy tasks), falling back to CPU for transcription preview")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
    except Exception:
        logger.warning(f"[{upload_id}] failed to inspect workers, proceeding on GPU")

    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    proc = prepare_preview_segment(upload_id)
    model = get_whisper_model()
    segments_gen, _ = model.transcribe(
        proc.stdout,
        word_timestamps=True,
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
        logger.info(f"[{upload_id}] GPU memory reserved before transcription: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

    # попытка авто-дауншифта на CPU если GPU перегружены
    try:
        inspector = app.control.inspect()
        active = inspector.active() or {}
        heavy = 0
        for node_tasks in active.values():
            for t in node_tasks:
                if t["name"] in ("tasks.diarize_full", "tasks.transcribe_segments") and t["name"] != "tasks.transcribe_segments":
                    heavy += 1
        if heavy >= 2 and self.request.delivery_info.get("routing_key") != "transcribe_cpu":
            logger.info(f"[{upload_id}] GPUs appear busy ({heavy} heavy tasks), rescheduling transcription to CPU")
            transcribe_segments.apply_async((upload_id, correlation_id), queue="transcribe_cpu")
            return
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

    # общие параметры декодера — чтобы совпадать по стилю с превью
    beam_size = int(getattr(settings, "WHISPER_BEAM_SIZE", 1))
    temperature = float(getattr(settings, "WHISPER_TEMPERATURE", 0.0))
    use_vad = bool(USE_VAD_IN_FULL)  # по умолчанию False, как в превью
    vad_min_sil_ms = int(getattr(settings, "VAD_MIN_SIL_MS", int(settings.SENTENCE_MAX_GAP_S * 1000)))
    vad_pad_ms = int(getattr(settings, "VAD_SPEECH_PAD_MS", 200))
    overlap = max(0.0, float(TRANSCRIBE_OVERLAP_S))
    chunk_len = float(getattr(settings, "CHUNK_LENGTH_S", 120.0))

    def _decode(source, start_shift: float = 0.0):
        kw = dict(
            word_timestamps=True,
            condition_on_previous_text=False,
            beam_size=beam_size,
            temperature=temperature,
        )
        if use_vad:
            kw.update(
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": vad_min_sil_ms,
                    "speech_pad_ms": vad_pad_ms,
                }
            )
        # язык опционально
        if settings.WHISPER_LANGUAGE:
            kw.update(language=settings.WHISPER_LANGUAGE)

        segs_gen, _ = model.transcribe(source, **kw)

        out = []
        for s in segs_gen:
            s.start += start_shift
            s.end += start_shift
            out.append(s)
        return out

    if duration <= max(chunk_len, 1.0):
        logger.info(f"[{upload_id}] short audio ({duration:.1f}s) → single pass (overlap={overlap}s)")
        raw_segs = _decode(str(wav), 0.0)
    else:
        processed_key = f"transcribe:processed_chunks:{upload_id}"
        processed = {int(x) for x in r.smembers(processed_key)}

        total_chunks = int(math.ceil((duration + overlap) / (chunk_len)))
        offset = 0.0
        chunk_idx = 0

        while offset < duration:
            # ядро чанка и его «пэды» на вход ffmpeg
            core_len = min(chunk_len, duration - offset)
            left_pad = overlap if offset > 0 else 0.0
            right_pad = overlap if (offset + core_len) < duration else 0.0

            # пропуск уже готовых
            if chunk_idx in processed:
                logger.info(f"[{upload_id}] skip chunk {chunk_idx+1}/{total_chunks} (already done)")
                offset += core_len
                chunk_idx += 1
                continue

            # фактический временной срез, который режем в pipe
            cut_start = max(0.0, offset - left_pad)
            cut_dur = min(duration, offset + core_len + right_pad) - cut_start

            logger.info(
                f"[{upload_id}] transcribe chunk {chunk_idx+1}/{total_chunks}: "
                f"cut {cut_start:.1f}s→{cut_start+cut_dur:.1f}s (core {offset:.1f}s→{offset+core_len:.1f}s, pad L={left_pad:.1f}s R={right_pad:.1f}s)"
            )

            try:
                p = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-threads", str(settings.FFMPEG_THREADS),
                        "-ss", str(cut_start),
                        "-t", str(cut_dur),
                        "-i", str(wav),
                        "-f", "wav", "pipe:1"
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )

                # декодируем и смещаем таймкоды к абсолютным
                segs = _decode(p.stdout, cut_start)
                p.stdout.close(); p.wait()

                # оставляем только сегменты, пересекающие ядро чанка
                core_start = offset
                core_end = offset + core_len
                for s in segs:
                    if s.end <= core_start or s.start >= core_end:
                        continue
                    # подрезаем к границам ядра, чтобы убрать влияние перекрытий
                    s.start = max(s.start, core_start)
                    s.end = min(s.end, core_end)
                    raw_segs.append(s)

                r.sadd(processed_key, chunk_idx)

            except Exception as e:
                logger.error(f"[{upload_id}] error in transcribe chunk {chunk_idx+1}/{total_chunks}: {e}", exc_info=True)
                try:
                    import torch; torch.cuda.empty_cache()
                except ImportError:
                    pass
            finally:
                offset += core_len
                chunk_idx += 1

        r.delete(processed_key)

    # плоский вид + группировка в предложения
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
        logger.info(f"[{upload_id}] GPU memory reserved after transcription: {torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}")
    except ImportError:
        pass

@app.task(bind=True, queue="diarize_gpu")
def diarize_full(self, upload_id, correlation_id):
    import gc
    r = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    logger.info(f"[{upload_id}] diarize_full started")
    r.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_started"}))
    deliver_webhook.delay("diarization_started", upload_id, None)

    # ---- GPU mem before
    try:
        import torch
        logger.info(
            f"[{upload_id}] GPU memory reserved before diarization: "
            f"{torch.cuda.memory_reserved() if torch.cuda.is_available() else 'n/a'}"
        )
    except ImportError:
        torch = None  # type: ignore

    wav, duration = prepare_wav(upload_id)

    # --- chunking params from env ---
    raw_chunk_limit = getattr(settings, "DIARIZATION_CHUNK_LENGTH_S", 0)
    try:
        chunk_limit = int(raw_chunk_limit)
    except Exception:
        logger.warning(f"[{upload_id}] invalid DIARIZATION_CHUNK_LENGTH_S={raw_chunk_limit!r}, falling back to 0")
        chunk_limit = 0

    pad = float(getattr(settings, "DIARIZATION_CHUNK_PADDING_S", 0.0) or 0.0)
    using_chunking = bool(chunk_limit and duration > chunk_limit)
    total_chunks = math.ceil(duration / chunk_limit) if using_chunking else 1

    # --- optional hints: exact / min-max speakers
    infer_kwargs: Dict[str, Any] = {}
    if getattr(settings, "PYANNOTE_NUM_SPEAKERS", None):
        try:
            infer_kwargs["num_speakers"] = int(getattr(settings, "PYANNOTE_NUM_SPEAKERS"))
        except Exception:
            pass
    else:
        if getattr(settings, "PYANNOTE_MIN_SPEAKERS", None):
            try:
                infer_kwargs["min_speakers"] = int(getattr(settings, "PYANNOTE_MIN_SPEAKERS"))
            except Exception:
                pass
        if getattr(settings, "PYANNOTE_MAX_SPEAKERS", None):
            try:
                infer_kwargs["max_speakers"] = int(getattr(settings, "PYANNOTE_MAX_SPEAKERS"))
            except Exception:
                pass

    logger.info(
        f"[{upload_id}] diarization plan: chunk_length={chunk_limit}s, "
        f"padding={pad}s, total_chunks={total_chunks}, infer_kwargs={infer_kwargs or '{}'}"
    )

    if not _PN_AVAILABLE:
        logger.error(f"[{upload_id}] pyannote.audio not available, aborting diarization")
        deliver_webhook.delay("processing_failed", upload_id, None)
        return

    pipeline = get_diarization_pipeline()
    raw: List[Dict[str, Any]] = []

    if using_chunking:
        processed_key = f"diarize:processed_chunks:{upload_id}"
        processed = {int(x) for x in r.smembers(processed_key)}

        offset = 0.0
        chunk_idx = 0
        while offset < duration:
            this_len = min(chunk_limit, duration - offset)
            left_pad = pad if offset > 0 else 0.0
            right_pad = pad if (offset + this_len) < duration else 0.0

            if chunk_idx in processed:
                logger.info(f"[{upload_id}] skip diarize chunk {chunk_idx+1}/{total_chunks}")
                offset += this_len
                chunk_idx += 1
                continue

            log_start = max(0.0, offset - left_pad)
            log_end = min(duration, offset + this_len + right_pad)
            logger.info(
                f"[{upload_id}] diarize chunk {chunk_idx+1}/{total_chunks}: "
                f"{log_start:.1f}s→{log_end:.1f}s (core {offset:.1f}s→{offset+this_len:.1f}s, pad L={left_pad:.1f}s R={right_pad:.1f}s)"
            )

            tmp = Path(settings.DIARIZER_CACHE_DIR) / f"{upload_id}_chunk_{chunk_idx}.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-threads", str(max(1, settings.FFMPEG_THREADS // 2)),
                    "-ss", str(log_start),
                    "-t", str(log_end - log_start),
                    "-i", str(wav),
                    str(tmp),
                ],
                check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )

            try:
                if torch:
                    with torch.inference_mode():
                        ann = pipeline(str(tmp), **infer_kwargs)
                else:
                    ann = pipeline(str(tmp), **infer_kwargs)

                before = len(raw)
                # перенос локальных координат в глобальные; обрезаем к ядру
                for s, _, spk in ann.itertracks(yield_label=True):
                    g_start = float(s.start) + log_start
                    g_end = float(s.end) + log_start
                    if g_end <= offset or g_start >= (offset + this_len):
                        continue
                    g_start = max(g_start, offset)
                    g_end = min(g_end, offset + this_len)
                    if g_end > g_start:
                        raw.append({"start": g_start, "end": g_end, "speaker": spk})
                added = len(raw) - before

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
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                offset += this_len
                chunk_idx += 1

        r.delete(processed_key)

    else:
        logger.info(f"[{upload_id}] Short audio or chunking disabled, single diarization pass")
        if torch:
            with torch.inference_mode():
                ann = pipeline(str(wav), **infer_kwargs)
        else:
            ann = pipeline(str(wav), **infer_kwargs)
        for s, _, spk in ann.itertracks(yield_label=True):
            raw.append({"start": float(s.start), "end": float(s.end), "speaker": spk})
        try:
            del ann
        except Exception:
            pass
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- базовая сортировка
    raw.sort(key=lambda x: x["start"])

    # ---------- Optional FS-EEND refinement ----------
    def _refine_with_fs_eend(_raw: List[Dict[str, Any]], _wav: Path) -> List[Dict[str, Any]]:
        try:
            if not bool(getattr(settings, "USE_FS_EEND", False)):
                logger.info(f"[{upload_id}] FS-EEND disabled → skipping refinement")
                return _raw

            model_id = getattr(settings, "FS_EEND_MODEL_PATH", None)
            if not model_id:
                logger.info(f"[{upload_id}] FS-EEND path not set → skipping refinement")
                return _raw

            from pyannote.audio import Pipeline as _TmpPipe
            logger.info(f"[{upload_id}] FS-EEND refinement started with {model_id}")
            if torch:
                with torch.inference_mode():
                    eend = _TmpPipe.from_pretrained(model_id, use_auth_token=settings.HUGGINGFACE_TOKEN,
                                                    cache_dir=settings.DIARIZER_CACHE_DIR,
                                                    device=getattr(settings, "FS_EEND_DEVICE", "cuda"))
                    ann = eend(str(_wav))
            else:
                eend = _TmpPipe.from_pretrained(model_id, use_auth_token=settings.HUGGINGFACE_TOKEN,
                                                cache_dir=settings.DIARIZER_CACHE_DIR,
                                                device=getattr(settings, "FS_EEND_DEVICE", "cuda"))
                ann = eend(str(_wav))

            refined: List[Dict[str, Any]] = []
            for s, _, spk in ann.itertracks(yield_label=True):
                refined.append({"start": float(s.start), "end": float(s.end), "speaker": spk})
            refined.sort(key=lambda x: x["start"])

            # простая метрика: доля времени с >=2 активными говорящими
            def _overlap_score(items: List[Dict[str, Any]]) -> float:
                if not items:
                    return 0.0
                step = 0.05
                t0 = min(x["start"] for x in items)
                t1 = max(x["end"] for x in items)
                pts = int((t1 - t0) / step) + 1
                cnt2 = 0
                for i in range(pts):
                    t = t0 + i * step
                    active = sum(1 for x in items if x["start"] <= t < x["end"])
                    if active >= 2:
                        cnt2 += 1
                return cnt2 / max(1, pts)

            base_ovl = _overlap_score(_raw)
            eend_ovl = _overlap_score(refined)
            logger.info(f"[{upload_id}] FS-EEND overlap score: base={base_ovl:.3f}, eend={eend_ovl:.3f}")

            if eend_ovl > base_ovl + 0.05:
                logger.info(f"[{upload_id}] FS-EEND accepted")
                return refined
            logger.info(f"[{upload_id}] FS-EEND rejected (kept base diarization)")
            return _raw

        except Exception as e:
            logger.warning(f"[{upload_id}] FS-EEND refinement failed: {e}")
            return _raw

    raw = _refine_with_fs_eend(raw, wav)

    # ---------- Light post-processing to stabilize segments ----------
    # 1) drop ultra-short blips (<0.20s)
    MIN_SEG = float(getattr(settings, "DIARIZATION_MIN_SEGMENT_S", 0.20))
    filtered = [s for s in raw if s["end"] - s["start"] >= MIN_SEG]
    dropped = len(raw) - len(filtered)
    if dropped:
        logger.info(f"[{upload_id}] dropped {dropped} ultra-short segments (<{MIN_SEG:.2f}s)")
    raw = filtered

    # 2) merge same-speaker segments with tiny gaps (<0.20s)
    GAP_MERGE = float(getattr(settings, "DIARIZATION_MERGE_GAP_S", 0.20))
    raw.sort(key=lambda x: (x["speaker"], x["start"]))
    merged: List[Dict[str, Any]] = []
    cur = None
    for seg in sorted(raw, key=lambda x: x["start"]):
        if cur and cur["speaker"] == seg["speaker"] and seg["start"] - cur["end"] <= GAP_MERGE:
            cur["end"] = max(cur["end"], seg["end"])
        else:
            if cur:
                merged.append(cur)
            cur = dict(seg)
    if cur:
        merged.append(cur)
    raw = merged

    # 3) финальная сортировка
    raw.sort(key=lambda x: x["start"])

    # ---- NEW post-fixes: перекрашивание «ошибочных» коротышей/сэндвичей
    raw = _fix_sandwich_flips(
        raw,
        max_len=float(getattr(settings, "DIARIZATION_FIX_SANDWICH_S", 1.0))
    )
    raw = _glue_stray_shorts(
        raw,
        max_len=float(getattr(settings, "DIARIZATION_GLUE_MAX_S", 0.6)),
        max_gap=float(getattr(settings, "DIARIZATION_GLUE_GAP_S", 0.15))
    )

    # ---------- Optional stitching across chunks ----------
    if SPEAKER_STITCH_ENABLED and using_chunking:
        raw = stitch_speakers(raw, wav, upload_id)
    else:
        logger.debug(f"[{upload_id}] skipping speaker stitching (using_chunking={using_chunking})")

    # ---------- Collapse contiguous same-speaker segments ----------
    diar_sentences: List[Dict[str, Any]] = []
    buf: Optional[Dict[str, Any]] = None
    for seg in raw:
        if buf and buf["speaker"] == seg["speaker"] and (seg["start"] - buf["end"]) < 0.10:
            buf["end"] = seg["end"]
        else:
            if buf:
                diar_sentences.append(buf)
            buf = dict(seg)
    if buf:
        diar_sentences.append(buf)

    # ---------- Save diarization ----------
    out = Path(settings.RESULTS_FOLDER) / upload_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "diarization.json").write_text(json.dumps(diar_sentences, ensure_ascii=False, indent=2))

    speakers = sorted({seg["speaker"] for seg in diar_sentences if seg.get("speaker") is not None})
    logger.info(
        f"[{upload_id}] diarization_done, total segments: {len(diar_sentences)}, "
        f"speakers_detected={len(speakers)}"
    )
    if torch and torch.cuda.is_available():
        logger.info(f"[{upload_id}] GPU memory reserved after diarization: {torch.cuda.memory_reserved()}")

    # ---------- «Склейка» спикеров к фразам транскрипта (перезаписываем transcript.json) ----------
    try:
        tx_path = out / "transcript.json"
        if tx_path.exists():
            try:
                sentences = json.loads(tx_path.read_text())
                if isinstance(sentences, list):
                    aligned = merge_speakers(
                        sentences,
                        diar_sentences,
                        pad=float(getattr(settings, "ALIGN_PAD_S", 0.2))
                    )
                    # ВАЖНО: сохраняем в ТОТ ЖЕ ФАЙЛ и в ТОЙ ЖЕ СХЕМЕ
                    tx_path.write_text(json.dumps(aligned, ensure_ascii=False, indent=2))
                    logger.info(f"[{upload_id}] transcript.json overwritten with speakers ({len(aligned)} sentences)")
            except Exception as e:
                logger.warning(f"[{upload_id}] failed to align transcript.json: {e}")
        else:
            logger.info(f"[{upload_id}] transcript.json not found; skipping alignment")
    except Exception as e:
        logger.warning(f"[{upload_id}] speaker-to-phrase alignment failed: {e}")

    # ---------- notify ----------
    r.publish(
        f"progress:{upload_id}",
        json.dumps({
            "status": "diarization_done",
            "segments": len(diar_sentences),
            "speakers": len(speakers),
            "total_chunks": total_chunks
        })
    )
    deliver_webhook.delay("diarization_completed", upload_id, {"diarization": diar_sentences})


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