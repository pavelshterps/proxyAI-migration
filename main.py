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
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, FileResponse
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
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config.settings import settings
from database import init_models, engine, get_db
from crud import (
    create_upload_record,
    get_upload_for_user,
    create_admin_user as crud_create_admin_user,
    list_users as crud_list_users,
    delete_user as crud_delete_user
)
from dependencies import get_current_user

# --- Фоновые задачи ---
from tasks import convert_to_wav_and_preview, diarize_full

app = FastAPI(title="proxyAI", version=settings.APP_VERSION)

# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    {"detail": "Too Many Requests"}, status_code=429
))

# --- CORS & TrustedHost ---
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
    idx = Path("static/index.html")
    if not idx.exists():
        raise HTTPException(404, "Not found")
    return HTMLResponse(idx.read_text(encoding="utf-8"))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico = Path("static/favicon.ico")
    if ico.exists():
        return FileResponse(str(ico))
    raise HTTPException(404, "favicon not found")

# --- StructLog ---
structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()

# --- Redis для прогресса ---
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

@app.on_event("startup")
async def on_startup():
    # Инициализация БД и директорий
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

# --- Health / Ready / Metrics ---
@app.get("/health", tags=["default"])
async def health():
    return {"status": "ok"}

@app.get("/ready", tags=["default"])
async def ready():
    return {"status": "ready"}

@app.get("/metrics", tags=["default"])
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

# --- SSE: прогресс обработки ---
@app.get("/events/{upload_id}", tags=["default"])
async def progress_events(
    upload_id: str,
    request: Request,
    api_key: str = Depends(get_api_key),
):
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

# --- Быстрый статус без SSE ---
@app.get("/status/{upload_id}", tags=["default"])
async def get_status(upload_id: str, api_key: str = Depends(get_api_key)):
    raw = await redis.get(f"progress:{upload_id}")
    if not raw:
        raise HTTPException(404, "status not found")
    return json.loads(raw)

# --- Загрузка файла ---
@app.post("/upload/", tags=["default"])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(None),
    file_url: str = Form(None),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    if not file and not file_url:
        raise HTTPException(400, "Нужно передать либо файл, либо file_url")

    upload_id = uuid.uuid4().hex
    ext = ""
    try:
        Path(settings.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        if file:
            data = await file.read()
            if not data:
                raise HTTPException(400, "File is empty")
            ext = Path(file.filename).suffix or ""
            dst = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
            dst.write_bytes(data)
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

    cid = x_correlation_id or uuid.uuid4().hex
    await redis.set(f"progress:{upload_id}", json.dumps({"status": "started"}))
    await redis.publish(f"progress:{upload_id}", json.dumps({"status": "started"}))

    convert_to_wav_and_preview.delay(upload_id, cid)
    return JSONResponse({"upload_id": upload_id}, headers={"X-Correlation-ID": cid})

# --- Собранные результаты для UI ---
@app.get("/results/{upload_id}", tags=["default"])
async def get_results(
    upload_id: str,
    api_key: str = Depends(get_api_key),
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

    # если диаризации нет — просто отдаём чистый транскрипт
    return {"results": transcript}

# --- Сырые транскрипты + диаризация ---
@app.get("/transcription/{upload_id}", tags=["default"])
async def get_full_transcript(
    upload_id: str,
    api_key: str = Depends(get_api_key),
):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    if not p.exists():
        raise HTTPException(404, "transcript.json not found")
    return json.loads(p.read_text(encoding="utf-8"))

@app.get("/transcription/{upload_id}/preview", tags=["default"])
async def get_preview_transcript(
    upload_id: str,
    api_key: str = Depends(get_api_key),
):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "preview_transcript.json"
    if not p.exists():
        raise HTTPException(404, "preview_transcript.json not found")
    return json.loads(p.read_text(encoding="utf-8"))

@app.get("/diarization/{upload_id}", tags=["default"])
async def get_diarization(
    upload_id: str,
    api_key: str = Depends(get_api_key),
):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    if not p.exists():
        raise HTTPException(404, "diarization.json not found")
    return json.loads(p.read_text(encoding="utf-8"))

# --- Ручной запуск полной диаризации ---
@app.post("/diarize/{upload_id}", tags=["default"])
async def request_diarization(
    upload_id: str,
    api_key: str = Depends(get_api_key),
):
    await redis.set(f"diarize_requested:{upload_id}", "1")
    await redis.publish(f"progress:{upload_id}", json.dumps({"status": "diarize_requested"}))
    diarize_full.delay(upload_id, None)
    return {"message": "diarization started"}

# --- Сохранение пользовательских меток спикеров ---
@app.post("/labels/{upload_id}", tags=["default"])
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    api_key: str = Depends(get_api_key),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    rec.label_mapping = mapping
    await db.commit()

    # возвращаем сразу «собранный» результат
    base = Path(settings.RESULTS_FOLDER) / upload_id
    transcript = json.loads((base / "transcript.json").read_text())
    raw = []
    if (base / "diarization.json").exists():
        raw = json.loads((base / "diarization.json").read_text())
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
    return {"results": merged}

# --- Админ: CRUD для пользователей ---
class AdminCreatePayload(BaseModel):
    name: str

admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)
async def verify_admin(key: str = Depends(admin_key_header)):
    if key != settings.ADMIN_API_KEY:
        raise HTTPException(403, "Invalid admin key")
    return key

@app.get("/admin/users", tags=["admin"], dependencies=[Depends(verify_admin)])
async def list_admin_users(db=Depends(get_db)):
    users = await crud_list_users(db)
    return [
        {
            "id": u.id,
            "name": u.name,
            "api_key": u.api_key,
            "is_admin": getattr(u, "is_admin", False)
        }
        for u in users
    ]

@app.post("/admin/users", tags=["admin"], dependencies=[Depends(verify_admin)])
async def create_admin_user(
    payload: AdminCreatePayload,
    db=Depends(get_db),
):
    new_api_key = uuid.uuid4().hex
    new = await crud_create_admin_user(db, payload.name, new_api_key)
    return {
        "id": new.id,
        "name": new.name,
        "api_key": new.api_key,
        "is_admin": getattr(new, "is_admin", False)
    }

@app.delete("/admin/users/{user_id}", tags=["admin"], dependencies=[Depends(verify_admin)])
async def delete_admin_user(user_id: int, db=Depends(get_db)):
    ok = await crud_delete_user(db, user_id)
    if not ok:
        raise HTTPException(404, "User not found")
    return {"message": "deleted"}