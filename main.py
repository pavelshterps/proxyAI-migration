# main.py

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
    Header, Depends, Request, Response, Body
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
from tasks import download_audio, preview_transcribe, diarize_full

# === Application lifecycle & retry on DB init ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Retry init_models при старте сервера до 5 раз."""
    for attempt in range(1, 6):
        try:
            await init_models(engine)
            log.info("Database models initialized", attempt=attempt)
            break
        except OSError as e:
            log.warning(
                "init_models failed, retrying",
                attempt=attempt,
                error=str(e)
            )
            await asyncio.sleep(2)
    else:
        log.error("init_models permanently failed after 5 attempts")
    yield

# === FastAPI & middleware setup ===

app = FastAPI(
    title="proxyAI",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# structlog configuration
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

# CORS & trusted hosts
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

# Prometheus metrics
HTTP_REQ_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path"]
)
HTTP_REQ_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["path"]
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return response

# === Health & readiness endpoints ===

@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    log.debug("Health check", path=request.url.path)
    return {"status": "ok", "version": app.version}

@app.get("/ready")
async def ready():
    log.debug("Readiness check")
    return {"status": "ready", "version": app.version}

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

@app.post("/upload/", dependencies=[Depends(get_current_user)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    """
    1) Сохраняем файл
    2) Создаём запись в БД
    3) Запускаем download_audio и preview_transcribe
    4) Публикуем статус 'started' в Redis
    """
    start_ts = time.time()
    cid = x_correlation_id or uuid.uuid4().hex
    log.info(
        "Upload endpoint called",
        user_id=current_user.id,
        correlation_id=cid,
        client=request.client.host
    )

    data = await file.read()
    if not data:
        log.error("Upload failed: empty file", correlation_id=cid)
        raise HTTPException(400, "File is empty")

    ext = Path(file.filename).suffix or ""
    upload_id = uuid.uuid4().hex
    dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
    dest.write_bytes(data)
    log.info(
        "File saved",
        upload_id=upload_id,
        path=str(dest),
        size_bytes=len(data),
        elapsed_s=(time.time() - start_ts)
    )

    # create DB record
    try:
        await create_upload_record(db, current_user.id, upload_id)
        log.info("DB record created", upload_id=upload_id, user_id=current_user.id)
    except Exception as e:
        log.warning(
            "Failed to create upload record",
            upload_id=upload_id,
            error=str(e)
        )

    # set external mapping
    await redis.set(f"external:{upload_id}", upload_id)
    log.debug("Set external ID in Redis", upload_id=upload_id)

    # dispatch Celery tasks
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)
    log.info(
        "Dispatched Celery tasks",
        upload_id=upload_id,
        tasks=["download_audio", "preview_transcribe"]
    )

    # publish initial progress
    state = {"status": "started", "preview": None}
    await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    log.info("Published initial progress", upload_id=upload_id, state=state)

    elapsed = time.time() - start_ts
    log.info("Upload handler completed", upload_id=upload_id, elapsed_s=elapsed)

    return JSONResponse(
        {"upload_id": upload_id, "external_id": upload_id},
        headers={"X-Correlation-ID": cid}
    )

# === Server-Sent Events for progress ===

@app.get("/events/{upload_id}")
async def progress_events(upload_id: str):
    """
    SSE stream прогресса обработки (Redis pub/sub).
    """
    log.info("Client subscribed to SSE", upload_id=upload_id)
    async def generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"progress:{upload_id}")
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg["type"] == "message":
                    log.debug("SSE: sending message to client", upload_id=upload_id, data=msg["data"])
                    yield f"data: {msg['data']}\n\n"
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(f"progress:{upload_id}")
            log.info("Client unsubscribed from SSE", upload_id=upload_id)
    return EventSourceResponse(generator())

# === Results endpoint ===

@app.get("/results/{upload_id}", summary="Get preview, transcript or diarization")
async def get_results(
    upload_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    log.info("Get results called", upload_id=upload_id, user_id=current_user.id)

    # 1) preview
    pd = await redis.get(f"preview_result:{upload_id}")
    if pd:
        pl = json.loads(pd)
        log.info("Returning preview from Redis", upload_id=upload_id, segments=len(pl["timestamps"]))
        return JSONResponse(content={"results": pl["timestamps"], "text": pl["text"]})

    # 2) full transcript
    tp = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    if tp.exists():
        data = json.loads(tp.read_text(encoding="utf-8"))
        log.info("Returning full transcript file", upload_id=upload_id, path=str(tp), segments=len(data))
        return JSONResponse(content={"results": data})

    # 3) diarization + user mapping
    dp = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    if dp.exists():
        segs = json.loads(dp.read_text(encoding="utf-8"))
        rec = await get_upload_for_user(db, current_user.id, upload_id)
        mapping = rec.label_mapping or {}
        for s in segs:
            s["speaker"] = mapping.get(str(s["speaker"]), s["speaker"])
        log.info("Returning diarization", upload_id=upload_id, segments=len(segs))
        return JSONResponse(content={"results": segs})

    log.warning("Results not ready", upload_id=upload_id)
    raise HTTPException(404, "Results not ready")

# === Trigger diarization manually ===

@app.post("/diarize/{upload_id}", summary="Request diarization")
async def request_diarization(
    upload_id: str,
    current_user=Depends(get_current_user)
):
    log.info("Diarization requested", upload_id=upload_id, user_id=current_user.id)
    # mark in Redis
    await redis.set(f"diarize_requested:{upload_id}", "1")
    state = json.loads(await redis.get(f"progress:{upload_id}") or "{}")
    state["diarize_requested"] = True
    await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    log.debug("Published diarize_requested flag", upload_id=upload_id, state=state)

    diarize_full.delay(upload_id, None)
    log.info("Launched diarize_full task", upload_id=upload_id)

    return JSONResponse({"message": "diarization started"})

# === Save speaker labels ===

@app.post("/labels/{upload_id}", summary="Save speaker labels")
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    log.info("Save labels called", upload_id=upload_id, user_id=current_user.id, mapping=mapping)
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    rec.label_mapping = mapping
    await db.commit()
    log.debug("Updated label_mapping in DB", upload_id=upload_id)

    # update local diarization file accordingly
    out = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    updated = []
    if out.exists():
        segs = json.loads(out.read_text(encoding="utf-8"))
        for s in segs:
            new_spk = mapping.get(str(s["speaker"]), s["speaker"])
            updated.append({"start": s["start"], "end": s["end"], "speaker": new_spk})
        out.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Rewrote diarization.json with new labels", path=str(out), segments=len(updated))
    else:
        log.warning("Diarization file not found for updating labels", path=str(out))

    return JSONResponse({"results": updated})

# === Include routers & mount static ===

app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")