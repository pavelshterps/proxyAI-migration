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
            log.warning("init_models failed, retrying", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    else:
        log.error("init_models permanently failed after 5 attempts")
    yield

# === FastAPI & middleware setup ===

app = FastAPI(title="proxyAI", version=settings.APP_VERSION, lifespan=lifespan)

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

async def get_api_key(
    x_api_key: str = Depends(api_key_header),
    api_key: str = Query(None)
):
    """
    Получаем API-ключ из HTTP-заголовка X-API-Key или из query-параметра api_key.
    """
    key = x_api_key or api_key
    if not key:
        log.warning("Missing API key in request", path=str(api_key))
        raise HTTPException(401, "Missing API Key")
    return key

# Prometheus metrics
HTTP_REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method","path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds","HTTP latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
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
                # disconnect detection
                if await request.is_disconnected():
                    log.info("Client disconnected before completion", upload_id=upload_id)
                    break

                # try to fetch a real message
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                now = time.time()

                if msg and msg["type"] == "message":
                    payload = msg["data"]
                    log.debug("SSE ▶ sending JSON", upload_id=upload_id, data=payload)
                    # correct SSE framing
                    yield f"data: {payload}\n\n"
                # heartbeat ping every second
                elif now - last_hb > 1.0:
                    yield ":\n\n"
                    last_hb = now

                await asyncio.sleep(0.1)

        except Exception as e:
            log.error("Error in SSE generator", upload_id=upload_id, error=str(e))
            # let client know
            yield f"data: {{\"status\":\"error\",\"message\":\"{str(e)}\"}}\n\n"
        finally:
            await pubsub.unsubscribe(f"progress:{upload_id}")
            log.info("Client unsubscribed from SSE", upload_id=upload_id)

    return EventSourceResponse(generator())

# === Results ===
# (остальные endpoints: /results, /diarize, /labels остаются без изменений —
# они у вас уже стабильно работают по итогам предыдущих тестов)

app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")