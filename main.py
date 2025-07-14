import time
import uuid
import json
import asyncio
from pathlib import Path

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI,
    UploadFile, File, HTTPException,
    Header, Depends, Request, Body, Query
)
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from database import init_models, engine, get_db
from crud import (
    create_upload_record,
    get_upload_for_user,
    create_admin_user as crud_create_admin_user
)
from dependencies import get_current_user
from tasks import preview_transcribe, transcribe_segments, diarize_full

# --- structlog setup ---
structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()

app = FastAPI(title="proxyAI", version=settings.APP_VERSION)

# --- rate limiter ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    {"detail": "Too Many Requests"}, status_code=429
))

# --- CORS and Host trust ---
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

# --- serve your SPA ---
# static JS/CSS under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# root -> index.html
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = Path("static/index.html")
    if not index_path.exists():
        raise HTTPException(404, "Not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

# optional favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    maybe = Path("static/favicon.ico")
    if maybe.exists():
        return StaticFiles(directory="static").lookup_path("favicon.ico")
    raise HTTPException(404)


@app.on_event("startup")
async def startup():
    # retry DB connect
    for attempt in range(5):
        try:
            await init_models(engine)
            log.info("DB connected", attempt=attempt)
            break
        except OSError as e:
            log.warning("DB init failed", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    # ensure dirs
    for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
        log.debug("Ensured dir", path=str(d))

# Redis for SSE + Celery
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(
    x_api_key: str = Depends(api_key_header),
    api_key: str = Query(None),
):
    key = x_api_key or api_key
    if not key:
        raise HTTPException(401, "Missing API Key")
    return key

# SSE endpoint
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

# Upload endpoint
@app.post("/upload")
@limiter.limit("10/minute")
async def upload(
    request: Request,  # needed for slowapi
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
    await redis.set(f"progress:{upload_id}", json.dumps({"status":"started"}))
    await redis.publish(f"progress:{upload_id}", json.dumps({"status":"started"}))

    preview_transcribe.delay(upload_id, cid)
    return JSONResponse({"upload_id": upload_id}, headers={"X-Correlation-ID": cid})

# Get results
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
            orig = next(
                (d["speaker"] for d in raw if d["start"] <= seg["start"] < d["end"]),
                None
            )
            speaker = mapping.get(str(orig), orig)
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "orig": orig,
                "speaker": speaker
            })
        return {"results": merged}
    return {"results": transcript}

# Diarization trigger
@app.post("/diarize/{upload_id}")
async def request_diarization(
    upload_id: str,
    current_user=Depends(get_current_user),
):
    await redis.set(f"diarize_requested:{upload_id}", "1")
    await redis.publish(f"progress:{upload_id}", json.dumps({"status":"diarize_requested"}))
    diarize_full.delay(upload_id, None)
    return {"message": "diarization started"}

# Save labels
@app.post("/labels/{upload_id}")
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    rec.label_mapping = mapping
    await db.commit()

    base = Path(settings.RESULTS_FOLDER) / upload_id
    transcript = json.loads((base/"transcript.json").read_text())
    raw = (base/"diarization.json").exists() and json.loads((base/"diarization.json").read_text()) or []
    merged = []
    for seg in transcript:
        orig = next(
            (d["speaker"] for d in raw if d["start"] <= seg["start"] < d["end"]),
            None
        )
        speaker = mapping.get(str(orig), orig)
        merged.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "orig": orig,
            "speaker": speaker
        })
    return {"results": merged}

# --- Admin endpoint ---
class AdminCreatePayload(BaseModel):
    name: str

admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

async def verify_admin_key(key: str = Depends(admin_key_header)):
    # теперь проверяем правильный параметр из settings
    if key != settings.ADMIN_API_KEY:
        raise HTTPException(403, "Invalid admin key")
    return key

@app.post("/admin/users", dependencies=[Depends(verify_admin_key)])
async def create_admin_user(
    payload: AdminCreatePayload,
    db=Depends(get_db),
):
    """
    Создать нового admin-пользователя.
    Требует заголовок X-Admin-Key == settings.ADMIN_API_KEY
    """
    # создаём пользователя (api_key сгенерируется автоматически)
    new_user = await crud_create_admin_user(db, payload.name)
    return {
        "id": new_user.id,
        "name": new_user.name,
        "api_key": new_user.api_key,
        "is_admin": getattr(new_user, "is_admin", False)
    }