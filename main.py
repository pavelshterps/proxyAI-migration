import time
import uuid
import json
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
from typing import Optional

import yt_dlp
import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI,
    UploadFile, File, Form, HTTPException,
    Header, Depends, Request, Query
)
from fastapi.responses import (
    JSONResponse, HTMLResponse, PlainTextResponse, FileResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sse_starlette.sse import EventSourceResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config.settings import settings
from database import init_models, engine, get_db
from crud import create_upload_record, get_upload_for_user
from dependencies import get_current_user
from tasks import convert_to_wav_and_preview, diarize_full, merge_speakers
from routes import router as api_router
from admin_routes import router as admin_router

app = FastAPI(title="proxyAI", version=settings.APP_VERSION)

# rate limiter
timer_limiter = Limiter(key_func=get_remote_address)
app.state.limiter = timer_limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse({"detail": "Too Many Requests"}, status_code=429))

# CORS & TrustedHost
app.add_middleware(CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST,
)

app.include_router(api_router)
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")

# logging
structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()

# Redis
redis = redis_async.from_url(settings.REDIS_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def is_direct_file(url: str) -> bool:
    """
    Определяем, является ли URL прямой ссылкой на аудио-файл по расширению.
    """
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix in {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}

async def get_api_key(x_api_key: Optional[str] = Depends(api_key_header), api_key: str = Query(None)):
    key = x_api_key or api_key
    if not key:
        raise HTTPException(401, "Missing API Key")
    return key

@app.on_event("startup")
async def on_startup():
    # инициализация БД
    for attempt in range(5):
        try:
            await init_models(engine)
            log.info("DB connected", attempt=attempt)
            break
        except OSError as e:
            log.warning("DB init failed", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    # создаём директории
    for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
        log.debug("Ensured dir", path=str(d))

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = Path("static/index.html")
    if not index.exists():
        raise HTTPException(404, "Not found")
    return HTMLResponse(index.read_text(encoding="utf-8"))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico = Path("static/favicon.ico")
    if ico.exists():
        return FileResponse(str(ico))
    raise HTTPException(404, "favicon not found")

@app.get("/health", tags=["default"])
async def health():
    return {"status": "ok"}

@app.get("/ready", tags=["default"])
async def ready():
    return {"status": "ready"}

@app.get("/metrics", tags=["default"])
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/events/{upload_id}", tags=["default"])
async def progress_events(upload_id: str, request: Request, api_key: str = Depends(get_api_key)):
    async def event_generator():
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
    return EventSourceResponse(event_generator())

@app.get("/status/{upload_id}", tags=["default"])
async def get_status(upload_id: str, api_key: str = Depends(get_api_key)):
    raw = await redis.get(f"progress:{upload_id}")
    if not raw:
        raise HTTPException(404, "status not found")
    return json.loads(raw)

@app.post("/upload/", tags=["default"])
@timer_limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(None),
    file_url: str = Form(None),
    x_correlation_id: Optional[str] = Header(None),
    api_key: str = Depends(get_api_key),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    # проверяем вход
    if not file and not file_url:
        raise HTTPException(400, "Нужно передать либо файл, либо file_url")

    upload_id = uuid.uuid4().hex
    try:
        Path(settings.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        base = Path(settings.UPLOAD_FOLDER)
        dst_base = base / upload_id

        if file:
            data = await file.read()
            if not data:
                raise HTTPException(400, "File is empty")
            ext = Path(file.filename).suffix or ""
            dst = dst_base.with_suffix(ext)
            dst.write_bytes(data)
        else:
            # URL загрузка
            if not is_direct_file(file_url):
                # любой не-прямой аудио-URL — через yt-dlp
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": str(dst_base) + ".%(ext)s",
                    "quiet": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(file_url, download=True)
                ext = info.get('ext', '')
                downloaded = dst_base.with_suffix(f".{ext}")
                dst = base / f"{upload_id}.{ext}"
                downloaded.rename(dst)
            else:
                # прямой файл по расширению
                parsed = urlparse(file_url)
                ext = Path(parsed.path).suffix or ""
                dst = base / f"{upload_id}{ext}"
                urlretrieve(file_url, str(dst))
    except HTTPException:
        raise
    except Exception as e:
        log.error("upload save error", error=str(e))
        raise HTTPException(500, f"Cannot save source: {e}")

    # сохраняем запись в БД
    try:
        await create_upload_record(db, current_user.id, upload_id)
    except Exception:
        log.warning("DB create failed", upload_id=upload_id)

    # начинаем процессинг
    cid = x_correlation_id or uuid.uuid4().hex
    await redis.set(f"progress:{upload_id}", json.dumps({"status": "started"}))
    await redis.publish(f"progress:{upload_id}", json.dumps({"status": "started"}))
    convert_to_wav_and_preview.delay(upload_id, cid)

    return JSONResponse({"upload_id": upload_id}, headers={"X-Correlation-ID": cid})

@app.get("/results/{upload_id}", tags=["default"])
async def get_results(
    upload_id: str,
    api_key: str = Depends(get_api_key),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
    pad: float = Query(0.2, description="padding seconds when matching diarization"),
    include_orig: bool = Query(False, description="return orig diarization labels too"),
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

        merged = merge_speakers(transcript, raw, pad=pad)
        for seg in merged:
            # если orig не задан, берём текущее поле speaker
            orig = seg.get("orig", seg.get("speaker"))
            seg["orig"] = orig
            seg["speaker"] = mapping.get(str(orig), orig)
            if not include_orig:
                seg.pop("orig", None)
        return {"results": merged}

    return {"results": transcript}

@app.post("/diarize/{upload_id}", tags=["default"])
async def request_diarization(upload_id: str, api_key: str = Depends(get_api_key)):
    await redis.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_requested"}))
    diarize_full.delay(upload_id, None)
    return JSONResponse({"message": "diarization started"})
