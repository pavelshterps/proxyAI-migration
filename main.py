import time
from pathlib import Path

import structlog
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config.settings import settings
from routes import router as api_router

# structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()

# Ensure directories exist
for d in (settings.upload_folder, settings.results_folder, settings.diarizer_cache_dir):
    Path(d).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="proxyAI", version="13.7.3")

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
async def health():
    return {"status": "ok", "version": app.version}

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/upload/")
async def upload(
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None)
):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    upload_id = file.filename  # or uuid4()
    dest = Path(settings.upload_folder) / upload_id
    dest.write_bytes(data)

    log_ctx = log.bind(correlation_id=x_correlation_id, upload_id=upload_id, size=len(data))
    log_ctx.info("upload accepted")

    from tasks import transcribe_segments, diarize_full
    transcribe_segments.delay(upload_id, correlation_id=x_correlation_id)
    diarize_full.delay(upload_id, correlation_id=x_correlation_id)

    return {"upload_id": upload_id}

app.include_router(api_router, tags=["proxyAI"])
app.mount("/static", StaticFiles(directory="static"), name="static")