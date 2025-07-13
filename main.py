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

# импортируем все 4 таски, чтобы .delay(...) не падал
from tasks import download_audio, preview_transcribe, transcribe_segments, diarize_full

# === Application lifecycle & retry on DB init ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    for attempt in range(1, 6):
        try:
            await init_models(engine)
            log.info("Database models initialized", attempt=attempt)
            break
        except OSError as e:
            log.warning("init_models failed, retrying", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    else:
        log.error("init_models permanently failed after 5 attempts")
    yield

# === FastAPI & middleware setup ===
app = FastAPI(title="proxyAI", version=settings.APP_VERSION, lifespan=lifespan)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# CORS & TrustedHost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*settings.ALLOWED_ORIGINS_LIST],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST
)

# ensure directories exist
for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)
    log.debug("Ensured directory exists", path=str(d))

# Redis & security
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(
    x_api_key: str = Depends(api_key_header),
    api_key: str = Query(None)
):
    key = x_api_key or api_key
    if not key:
        log.warning("Missing API key in request")
        raise HTTPException(401, "Missing API Key")
    return key

# Prometheus metrics
HTTP_REQ_COUNT   = Counter("http_requests_total", "Total HTTP requests", ["method","path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds","HTTP latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    resp  = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return resp

# === Health & readiness ===
@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    log.debug("Health check", path=request.url.path)
    return {"status":"ok","version":app.version}

@app.get("/ready")
async def ready():
    log.debug("Readiness check")
    return {"status":"ready","version":app.version}

@app.get("/metrics")
@limiter.limit("10/minute")
async def metrics_endpoint(request: Request):
    log.debug("Metrics endpoint called")
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/", include_in_schema=False)
async def root():
    log.debug("Serving index.html")
    return FileResponse("static/index.html")

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
    start_ts = time.time()
    cid = x_correlation_id or uuid.uuid4().hex
    log.info("Upload endpoint called",
             user_id=current_user.id,
             correlation_id=cid,
             client=request.client.host)

    data = await file.read()
    if not data:
        log.error("Upload failed: empty file", correlation_id=cid)
        raise HTTPException(400, "File is empty")

    ext = Path(file.filename).suffix or ""
    upload_id = uuid.uuid4().hex
    dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
    dest.write_bytes(data)
    log.info("File saved",
             upload_id=upload_id,
             path=str(dest),
             size_bytes=len(data),
             elapsed_s=(time.time() - start_ts))

    # create DB record
    try:
        await create_upload_record(db, current_user.id, upload_id)
        log.info("DB record created", upload_id=upload_id, user_id=current_user.id)
    except Exception as e:
        log.warning("Failed to create upload record",
                    upload_id=upload_id,
                    error=str(e))

    # set external mapping
    await redis.set(f"external:{upload_id}", upload_id)
    log.debug("Set external ID in Redis", upload_id=upload_id)

    # dispatch Celery tasks
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)
    log.info("Dispatched Celery tasks",
             upload_id=upload_id,
             tasks=["download_audio", "preview_transcribe"])

    # publish initial progress
    state = {"status":"started","preview":None}
    await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    log.info("Published initial progress", upload_id=upload_id, state=state)

    return JSONResponse(
        {"upload_id":upload_id,"external_id":upload_id},
        headers={"X-Correlation-ID":cid}
    )

# === Server-Sent Events for progress ===
@app.get("/events/{upload_id}")
async def progress_events(
    upload_id: str,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    log.info("Client subscribed to SSE", upload_id=upload_id)

    async def generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"progress:{upload_id}")
        last_hb = time.time()

        try:
            while True:
                if await request.is_disconnected():
                    log.info("Client disconnected before completion", upload_id=upload_id)
                    break

                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                now = time.time()

                if msg and msg["type"] == "message":
                    payload = msg["data"]
                    log.debug("SSE ▶ sending JSON", upload_id=upload_id, data=payload)
                    yield f"data: {payload}\n\n"
                elif now - last_hb > 1.0:
                    yield ":\n\n"
                    last_hb = now

                await asyncio.sleep(0.1)
        except Exception as e:
            log.error("Error in SSE generator", upload_id=upload_id, error=str(e))
            yield f"data: {{\"status\":\"error\",\"message\":\"{str(e)}\"}}\n\n"
        finally:
            await pubsub.unsubscribe(f"progress:{upload_id}")
            log.info("Client unsubscribed from SSE", upload_id=upload_id)

    return EventSourceResponse(generator())

# === Results endpoint ===
@app.get("/results/{upload_id}", summary="Get preview, transcript and speaker labels")
async def get_results(
    upload_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    log.info("Get results called", upload_id=upload_id, user_id=current_user.id)

    base = Path(settings.RESULTS_FOLDER) / upload_id

    # 1) preview
    preview_file = base / "preview.json"
    if preview_file.exists():
        pl = json.loads(preview_file.read_text(encoding="utf-8"))
        return JSONResponse(content={"results": pl["timestamps"], "text": pl["text"]})

    # 2) full transcript
    tp = base / "transcript.json"
    if not tp.exists():
        log.warning("Results not ready", upload_id=upload_id)
        raise HTTPException(404, "Results not ready")

    transcript = json.loads(tp.read_text(encoding="utf-8"))

    # 3) merge with diarization if exists
    dp = base / "diarization.json"
    if dp.exists():
        diar = json.loads(dp.read_text(encoding="utf-8"))

        # apply user labels
        rec = await get_upload_for_user(db, current_user.id, upload_id)
        label_map = rec.label_mapping or {}
        for seg in diar:
            seg["speaker"] = label_map.get(str(seg["speaker"]), seg["speaker"])

        merged = []
        for seg in transcript:
            spk = next(
                (d["speaker"] for d in diar
                 if d["start"] <= seg["start"] < d["end"]),
                None
            )
            merged.append({
                "start":   seg["start"],
                "end":     seg["end"],
                "text":    seg["text"],
                "speaker": spk
            })

        log.info("Returning merged transcript + speakers", upload_id=upload_id, segments=len(merged))
        return JSONResponse(content={"results": merged})

    # 4) only transcript
    log.info("Returning transcript only", upload_id=upload_id, segments=len(transcript))
    return JSONResponse(content={"results": transcript})

# === Trigger diarization manually ===
@app.post("/diarize/{upload_id}", summary="Request diarization")
async def request_diarization(
    upload_id: str,
    current_user=Depends(get_current_user)
):
    log.info("Diarization requested", upload_id=upload_id, user_id=current_user.id)
    try:
        await redis.set(f"diarize_requested:{upload_id}", "1")
        state = json.loads(await redis.get(f"progress:{upload_id}") or "{}")
        state["diarize_requested"] = True
        await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
        log.debug("Published diarize_requested flag", upload_id=upload_id, state=state)

        diarize_full.delay(upload_id, None)
        log.info("Launched diarize_full task", upload_id=upload_id)
        return JSONResponse({"message": "diarization started"})
    except Exception as e:
        log.error("Failed to launch diarization", upload_id=upload_id, error=str(e))
        raise HTTPException(500, f"Diarize launch failed: {e}")

# === Save speaker labels ===
@app.post("/labels/{upload_id}", summary="Save speaker labels")
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    log.info("Save labels called", upload_id=upload_id, user_id=current_user.id, mapping=mapping)

    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        log.error("Upload not found for saving labels", upload_id=upload_id, user_id=current_user.id)
        raise HTTPException(404, "upload_id not found")

    rec.label_mapping = mapping
    await db.commit()
    log.debug("Updated label_mapping in DB", upload_id=upload_id)

    base = Path(settings.RESULTS_FOLDER) / upload_id
    dfile = base / "diarization.json"
    updated = []
    if dfile.exists():
        diar = json.loads(dfile.read_text(encoding="utf-8"))
        for seg in diar:
            new_spk = mapping.get(str(seg["speaker"]), seg["speaker"])
            updated.append({
                "start": seg["start"],
                "end":   seg["end"],
                "speaker": new_spk
            })
        dfile.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("Rewrote diarization.json with new labels", path=str(dfile), segments=len(updated))
    else:
        log.warning("Diarization file not found for updating labels", path=str(dfile))

    # return merged immediately
    merged = []
    for seg in json.loads((base / "transcript.json").read_text(encoding="utf-8")):
        spk = next(
            (d["speaker"] for d in updated
             if d["start"] <= seg["start"] < d["end"]),
            None
        )
        merged.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg["text"],
            "speaker": spk
        })
    log.info("Returning merged transcript + speakers after labels", upload_id=upload_id, segments=len(merged))
    return JSONResponse(content={"results": merged})

# === Include routers & mount static ===
app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")