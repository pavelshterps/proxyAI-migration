import time
import uuid
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    Header, Depends, Request, Response, Body, Query
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from database import get_db, engine, init_models
from crud import create_upload_record, get_upload_for_user
from dependencies import get_current_user
from routes import router as api_router
from admin_routes import router as admin_router

# импорт тасков
from tasks import download_audio, preview_transcribe, transcribe_segments, diarize_full

# === Lifecycle & FastAPI setup ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    for i in range(5):
        try:
            await init_models(engine)
            log.info("DB initialized", attempt=i+1)
            break
        except OSError as e:
            log.warning("DB init failed, retry", attempt=i+1, error=str(e))
            await asyncio.sleep(2)
    yield

app = FastAPI(title="proxyAI", version=settings.APP_VERSION, lifespan=lifespan)
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# Middlewares
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
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
    allowed_hosts=["127.0.0.1","localhost"] + settings.ALLOWED_ORIGINS_LIST
)

# Ensure dirs
for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)
    log.debug("Ensured dir exists", path=str(d))

# Redis & API-Key
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def get_api_key(x_api_key: str = Depends(api_key_header), api_key: str = Query(None)):
    key = x_api_key or api_key
    if not key:
        raise HTTPException(401, "Missing API Key")
    return key

# Metrics middleware (omitted for brevity)...

# === Upload endpoint ===
@app.post("/upload", dependencies=[Depends(get_current_user)])
@app.post("/upload/", dependencies=[Depends(get_current_user)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    cid = x_correlation_id or uuid.uuid4().hex
    data = await file.read()
    if not data:
        raise HTTPException(400, "File is empty")

    ext = Path(file.filename).suffix or ""
    upload_id = uuid.uuid4().hex
    dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
    dest.write_bytes(data)

    try:
        await create_upload_record(db, current_user.id, upload_id)
    except Exception:
        log.warning("DB record create failed", upload_id=upload_id)

    await redis.set(f"external:{upload_id}", upload_id)
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)

    state = {"status":"started","preview":None}
    await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    return JSONResponse({"upload_id":upload_id,"external_id":upload_id}, headers={"X-Correlation-ID":cid})

# === SSE endpoint === (unchanged)...

# === Results endpoint ===
@app.get("/results/{upload_id}", summary="Get transcript + speaker labels")
async def get_results(upload_id: str, current_user=Depends(get_current_user), db=Depends(get_db)):
    base = Path(settings.RESULTS_FOLDER) / upload_id

    # full transcript exists?
    tp = base / "transcript.json"
    if tp.exists():
        transcript = json.loads(tp.read_text(encoding="utf-8"))

        dp = base / "diarization.json"
        if dp.exists():
            raw_diar = json.loads(dp.read_text(encoding="utf-8"))
            rec = await get_upload_for_user(db, current_user.id, upload_id)
            label_map = rec.label_mapping or {}

            merged = []
            for seg in transcript:
                orig = next((d["speaker"] for d in raw_diar
                             if d["start"] <= seg["start"] < d["end"]), None)
                speaker = label_map.get(str(orig), orig)
                merged.append({
                    "start":   seg["start"],
                    "end":     seg["end"],
                    "text":    seg["text"],
                    "orig":    orig,      # неизменяемый ключ
                    "speaker": speaker   # текущее имя
                })

            return JSONResponse({"results": merged})

        # без diarization
        return JSONResponse({"results": transcript})

    # preview
    pf = base / "preview.json"
    if pf.exists():
        pl = json.loads(pf.read_text(encoding="utf-8"))
        return JSONResponse({"results": pl["timestamps"], "text": pl["text"]})

    raise HTTPException(404, "Results not ready")

# === Trigger diarization === (unchanged)...

# === Save speaker labels ===
@app.post("/labels/{upload_id}", summary="Save speaker labels")
async def save_labels(upload_id: str, mapping: dict = Body(...),
                      current_user=Depends(get_current_user), db=Depends(get_db)):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload not found")

    rec.label_mapping = mapping
    await db.commit()

    base = Path(settings.RESULTS_FOLDER) / upload_id
    raw_dfile = base / "diarization.json"
    if not raw_dfile.exists():
        raise HTTPException(404, "diarization not found")

    raw_diar = json.loads(raw_dfile.read_text(encoding="utf-8"))
    # перезаписываем файл при желании:
    updated = []
    for seg in raw_diar:
        new_spk = mapping.get(str(seg["speaker"]), seg["speaker"])
        updated.append({"start":seg["start"],"end":seg["end"],"speaker":new_spk})
    raw_dfile.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    # возвращаем merged сразу
    transcript = json.loads((base / "transcript.json").read_text(encoding="utf-8"))
    merged = []
    for seg in transcript:
        orig = next((d["speaker"] for d in raw_diar
                     if d["start"] <= seg["start"] < d["end"]), None)
        speaker = mapping.get(str(orig), orig)
        merged.append({
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
            "orig":    orig,
            "speaker": speaker
        })

    return JSONResponse({"results": merged})

# === Routers & static mount ===
app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")