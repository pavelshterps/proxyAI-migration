# main.py
import time
import uuid
import json
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI,
    UploadFile, File, Form, HTTPException,
    Header, Depends, Request, Body, Query
)
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
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

# --- Фоновые задачи ---
from tasks import convert_to_wav_and_preview, transcribe_segments, diarize_full

# --- StructLog ---
structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()

app = FastAPI(title="proxyAI", version=settings.APP_VERSION)

# --- Rate limiter & middleware ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse({"detail": "Too Many Requests"}, status_code=429)
)

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

# --- Статика и SPA ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = Path("static/index.html")
    if not index.exists():
        raise HTTPException(404, "Not found")
    return HTMLResponse(index.read_text(encoding="utf-8"))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico_path = Path("static/favicon.ico")
    if ico_path.exists():
        return FileResponse(str(ico_path))
    raise HTTPException(404, "favicon not found")

@app.on_event("startup")
async def on_startup():
    # DB + dirs
    for attempt in range(5):
        try:
            await init_models(engine)
            log.info("DB connected", attempt=attempt)
            break
        except OSError as e:
            log.warning("DB init failed", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    for d in (
        settings.UPLOAD_FOLDER,
        settings.RESULTS_FOLDER,
        settings.DIARIZER_CACHE_DIR
    ):
        Path(d).mkdir(parents=True, exist_ok=True)
        log.debug("Ensured dir", path=str(d))

# --- Redis для прогресса и API-Key ---
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

# --- 1) SSE событий ---
@app.get("/events/{upload_id}", dependencies=[Depends(get_api_key)])
async def progress_events(upload_id: str, request: Request):
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

# --- 2) Ручной upload (10/min) ---
@app.post("/upload/", dependencies=[Depends(get_api_key)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(None),
    file_url: str = Form(None),
    x_corr: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    if not file and not file_url:
        raise HTTPException(400, "Нужно передать либо file, либо file_url")

    upload_id = uuid.uuid4().hex
    try:
        Path(settings.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        if file:
            data = await file.read()
            if not data:
                raise HTTPException(400, "File is empty")
            ext = Path(file.filename).suffix or ""
            (Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}").write_bytes(data)
        else:
            parsed = urlparse(file_url)
            ext = Path(parsed.path).suffix or ""
            dst = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
            urlretrieve(file_url, str(dst))
    except HTTPException:
        raise
    except Exception as e:
        log.error("upload save error", error=str(e))
        raise HTTPException(500, f"Cannot save source: {e}")

    try:
        await create_upload_record(db, current_user.id, upload_id)
    except:
        log.warning("DB create failed", upload_id=upload_id)

    corr_id = x_corr or uuid.uuid4().hex
    await redis.set(f"progress:{upload_id}", json.dumps({"status": "started"}))
    await redis.publish(f"progress:{upload_id}", json.dumps({"status": "started"}))

    convert_to_wav_and_preview.delay(upload_id, corr_id)
    return JSONResponse({"upload_id": upload_id}, headers={"X-Correlation-ID": corr_id})

# --- 3) Результаты (UI-сборка) ---
@app.get("/results/{upload_id}", dependencies=[Depends(get_api_key)])
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

    # если есть diarization.json — мержим спикеров
    dp = base / "diarization.json"
    if dp.exists():
        raw = json.loads(dp.read_text(encoding="utf-8"))
        rec = await get_upload_for_user(db, current_user.id, upload_id)
        mapping = rec.label_mapping or {}
        merged = []
        for seg in transcript:
            orig = next(
                (d["speaker"] for d in raw
                 if d["start"] <= seg["start"] < d["end"]),
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

# --- 4) Ручная диаризация ---
@app.post("/diarize/{upload_id}", dependencies=[Depends(get_api_key)])
async def request_diarization(
    upload_id: str,
    current_user=Depends(get_current_user),
):
    await redis.set(f"diarize_requested:{upload_id}", "1")
    await redis.publish(
        f"progress:{upload_id}",
        json.dumps({"status": "diarize_requested"})
    )
    diarize_full.delay(upload_id, None)
    return {"message": "diarization started"}

# --- 5) Сырые транскрипты/диаризации (если нужно) ---
@app.get("/transcription/{upload_id}/preview", dependencies=[Depends(get_api_key)])
async def get_preview_transcript(upload_id: str):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "preview_transcript.json"
    if not p.exists():
        raise HTTPException(404, "preview not found")
    return json.loads(p.read_text(encoding="utf-8"))

@app.get("/transcription/{upload_id}", dependencies=[Depends(get_api_key)])
async def get_full_transcript(upload_id: str):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    if not p.exists():
        raise HTTPException(404, "transcript not found")
    return json.loads(p.read_text(encoding="utf-8"))

@app.get("/diarization/{upload_id}", dependencies=[Depends(get_api_key)])
async def get_diarization(upload_id: str):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    if not p.exists():
        raise HTTPException(404, "diarization not found")
    return json.loads(p.read_text(encoding="utf-8"))

# --- 6) Сохранение меток спикеров ---
@app.post("/labels/{upload_id}", dependencies=[Depends(get_api_key)])
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
    transcript = json.loads((base / "transcript.json").read_text())
    raw = []
    if (base / "diarization.json").exists():
        raw = json.loads((base / "diarization.json").read_text())
    merged = []
    for seg in transcript:
        orig = next(
            (d["speaker"] for d in raw
             if d["start"] <= seg["start"] < d["end"]),
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

# --- Admin: создание пользователей ---
class AdminCreatePayload(BaseModel):
    name: str

admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)
async def verify_admin_key(key: str = Depends(admin_key_header)):
    if key != settings.ADMIN_API_KEY:
        raise HTTPException(403, "Invalid admin key")
    return key

@app.post("/admin/users", dependencies=[Depends(verify_admin_key)])
async def create_admin_user(
    payload: AdminCreatePayload,
    db=Depends(get_db),
):
    new_key = uuid.uuid4().hex
    u = await crud_create_admin_user(db, payload.name, new_key)
    return {
        "id": u.id,
        "name": u.name,
        "api_key": u.api_key,
        "is_admin": getattr(u, "is_admin", False)
    }