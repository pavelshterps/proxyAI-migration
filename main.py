import time
import uuid
from pathlib import Path

import structlog
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config.settings import settings
from routes import router as api_router

# structlog JSON
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# rate limiter: 20 req/min per IP
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="proxyAI", version="13.7.6.2")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Ensure directories exist
for d in (settings.upload_folder, settings.results_folder, settings.diarizer_cache_dir):
    Path(d).mkdir(parents=True, exist_ok=True)

# Host validation
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.allowed_origins
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP metrics
HTTP_REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_ts = time.time()
    response = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start_ts)
    return response

@app.get("/health")
@limiter.limit("30/minute")
async def health():
    return {"status": "ok", "version": app.version}

@app.get("/ready")
async def ready():
    # Kubernetes readiness probe
    return {"status": "ready", "version": app.version}

@app.get("/metrics")
@limiter.limit("10/minute")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/upload/")
@limiter.limit("10/minute")
async def upload(
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None)
):
    cid = x_correlation_id or str(uuid.uuid4())

    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    upload_id = file.filename
    dest = Path(settings.upload_folder) / upload_id
    dest.write_bytes(data)

    log.bind(correlation_id=cid, upload_id=upload_id, size=len(data)).info("upload accepted")

    from tasks import transcribe_segments, diarize_full
    transcribe_segments.delay(upload_id, correlation_id=cid)
    diarize_full.delay(upload_id, correlation_id=cid)

    return Response(
        content={"upload_id": upload_id},
        headers={"X-Correlation-ID": cid}
    )

app.include_router(api_router, tags=["proxyAI"])
app.mount("/static", StaticFiles(directory="static"), name="static")