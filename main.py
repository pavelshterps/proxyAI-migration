import time
import uuid
import json
import asyncio
from pathlib import Path

import structlog
import redis.asyncio as redis_async
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Request, Body, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from database import init_models, engine, get_db
from crud import create_upload_record, get_upload_for_user
from dependencies import get_current_user
from tasks import download_audio, preview_transcribe, transcribe_segments, diarize_full

# --- structlog ---
structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()

# --- FastAPI & rate limiter setup ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="proxyAI", version=settings.APP_VERSION)

# Привязываем Limiter к состоянию приложения и инициализируем
app.state.limiter = limiter
limiter.init_app(app)

# Middleware
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST,
)

# Static files (если нужны)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Startup event: БД и папки ---
@app.on_event("startup")
async def startup():
    for attempt in range(5):
        try:
            await init_models(engine)
            log.info("DB connected", attempt=attempt)
            break
        except OSError as e:
            log.warning("DB init failed", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
        log.debug("Ensured dir", path=str(d))

# --- Redis для SSE ---
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# --- API Key dependency ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def get_api_key(x_api_key: str = Depends(api_key_header), api_key: str = Query(None)):
    key = x_api_key or api_key
    if not key:
        raise HTTPException(401, "Missing API Key")
    return key

# --- SSE endpoint ---
@app.get("/events/{upload_id}")
async def progress_events(
    upload_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
):
    async def gen():
        sub = redis.pubsub()
        await sub.subscribe(f"progress:{upload_id}")
        last_hb = time.time()
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await sub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                now = time.time()
                if msg and msg["type"] == "message":
                    yield f"data: {msg['data']}\n\n"
                elif now - last_hb > 1.0:
                    yield ":\n\n"
                    last_hb = now
                await asyncio.sleep(0.1)
        finally:
            await sub.unsubscribe(f"progress:{upload_id}")
    return EventSourceResponse(gen())

# --- Upload endpoint with rate limit ---
@app.post("/upload")
@limiter.limit("10/minute")
async def upload(
    request: Request,  # обязательно первым для slowapi
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "File is empty")
    ext = Path(file.filename).suffix or ""
    upload_id = uuid.uuid4().hex
    Path(settings.UPLOAD_FOLDER).joinpath(f"{upload_id}{ext}").write_bytes(data)
    try:
        await create_upload_record(db, current_user.id, upload_id)
    except Exception:
        log.warning("DB create failed", upload_id=upload_id)
    cid = x_correlation_id or uuid.uuid4().hex
    # Инициализируем прогресс
    await redis.set(f"progress:{upload_id}", json.dumps({"status":"started"}, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps({"status":"started"}, ensure_ascii=False))
    # Запускаем задачи
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)
    return JSONResponse({"upload_id": upload_id}, headers={"X-Correlation-ID": cid})

# --- Получение результатов ---
@app.get("/results/{upload_id}")
async def get_results(
    upload_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    base = Path(settings.RESULTS_FOLDER) / upload_id
    tp = base / "transcript.json"
    if not tp.exists():
        raise HTTPException(404, "Not ready")
    transcript = json.loads(tp.read_text(encoding="utf-8"))
    dp = base / "diarization.json"
    if dp.exists():
        raw = json.loads(dp.read_text(encoding="utf-8"))
        rec = await get_upload_for_user(db, current_user.id, upload_id)
        mapping = rec.label_mapping or {}
        merged = []
        for seg in transcript:
            orig = next((d["speaker"] for d in raw if d["start"] <= seg["start"] < d["end"]), None)
            speaker = mapping.get(str(orig), orig)
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "orig": orig,
                "speaker": speaker
            })
        return JSONResponse({"results": merged})
    return JSONResponse({"results": transcript})

# --- Запрос диаризации вручную ---
@app.post("/diarize/{upload_id}")
async def request_diarization(
    upload_id: str,
    current_user=Depends(get_current_user),
):
    await redis.set(f"diarize_requested:{upload_id}", "1")
    await redis.publish(
        f"progress:{upload_id}",
        json.dumps({"status": "diarize_requested"}, ensure_ascii=False)
    )
    diarize_full.delay(upload_id, None)
    return JSONResponse({"message": "diarization started"})

# --- Сохранение меток спикеров ---
@app.post("/labels/{upload_id}")
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    rec.label_mapping = mapping
    await db.commit()
    # Возвращаем обновлённые результаты сразу
    base = Path(settings.RESULTS_FOLDER) / upload_id
    tp = base / "transcript.json"
    dp = base / "diarization.json"
    transcript = json.loads(tp.read_text(encoding="utf-8"))
    raw = dp.exists() and json.loads(dp.read_text(encoding="utf-8")) or []
    merged = []
    for seg in transcript:
        orig = next((d["speaker"] for d in raw if d["start"] <= seg["start"] < d["end"]), None)
        speaker = mapping.get(str(orig), orig)
        merged.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "orig": orig,
            "speaker": speaker
        })
    return JSONResponse({"results": merged})