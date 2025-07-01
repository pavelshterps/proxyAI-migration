import time
import uuid
from pathlib import Path

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Response, Header,
    Depends, WebSocket, status
)
import asyncio
from sqlalchemy.ext.asyncio import AsyncEngine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db, engine
from crud import (
    get_user_by_api_key,
    create_upload_record,
    get_upload_for_user
)
from models import Base
from routes import router as api_router
from dependencies import get_current_user
from admin_routes import router as admin_router

# создаём таблицы
async def init_models(async_engine: AsyncEngine):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="proxyAI", version="13.8.5.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.include_router(admin_router)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Redis Pub/Sub + key/value
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# API-Key → User
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Ensure dirs
for d in (
    settings.UPLOAD_FOLDER,
    settings.RESULTS_FOLDER,
    settings.DIARIZER_CACHE_DIR
):
    Path(d).mkdir(parents=True, exist_ok=True)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTTP_REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])

@app.on_event("startup")
async def on_startup():
    await init_models(engine)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return response

@app.get("/health")
@limiter.limit("30/minute")
async def health(request):
    return {"status": "ok", "version": app.version}

@app.get("/ready")
async def ready():
    return {"status": "ready", "version": app.version}

@app.get("/metrics")
@limiter.limit("10/minute")
async def metrics(request):
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get(
    "/status/{upload_id}",
    summary="Check processing status",
    responses={401: {"description": "Invalid X-API-Key"}, 404: {"description": "Not found"}}
)
async def get_status(
    upload_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(status_code=404, detail="upload_id not found")

    base = Path(settings.RESULTS_FOLDER) / upload_id
    done = (base / "transcript.json").exists() and (base / "diarization.json").exists()
    if done:
        status_str = "done"
    else:
        uploaded = (Path(settings.UPLOAD_FOLDER) / upload_id).exists()
        status_str = "processing" if uploaded else "queued"

    progress = await redis.get(f"progress:{upload_id}") or "0%"
    return {"status": status_str, "progress": progress}

@app.post(
    "/upload/",
    dependencies=[Depends(get_current_user)],
    responses={401: {"description": "Invalid X-API-Key"}}
)
@limiter.limit("10/minute")
async def upload(
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    cid = x_correlation_id or str(uuid.uuid4())

    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    if len(data) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    upload_id = file.filename
    dest = Path(settings.UPLOAD_FOLDER) / upload_id
    dest.write_bytes(data)

    await create_upload_record(db, current_user.id, upload_id)

    log.bind(
        correlation_id=cid,
        upload_id=upload_id,
        user_id=current_user.id
    ).info("upload accepted")

    from tasks import transcribe_segments, diarize_full
    transcribe_segments.delay(upload_id, correlation_id=cid)
    diarize_full.delay(upload_id, correlation_id=cid)

    await redis.publish(f"progress:{upload_id}", "0%")
    await redis.set(f"progress:{upload_id}", "0%")

    return Response(
        content={"upload_id": upload_id},
        headers={"X-Correlation-ID": cid}
    )

app.include_router(
    api_router,
    tags=["proxyAI"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")