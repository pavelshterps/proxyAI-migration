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

# Импортируем только таски — теперь download_audio гарантированно там есть
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

app = FastAPI(title="proxyAI", version=settings.APP_VERSION, lifespan=lifespan)

# structlog
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

# ensure directories
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

# Prometheus
HTTP_REQ_COUNT   = Counter("http_requests_total", "Total HTTP requests", ["method","path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds","HTTP latency", ["path"])
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    resp  = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return resp

# Health & readiness
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

# … дальше без изменений: upload, SSE, results, diarize, labels …

app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")