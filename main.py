import os
import time

import structlog
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from tasks import transcribe_segments, diarize_full
from config.settings import settings

# setup structlog for JSON output
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()

app = FastAPI(title="proxyAI")

# metrics
REQUESTS = Counter("proxyai_http_requests_total", "HTTP requests", ["method", "endpoint"])
LATENCY = Histogram("proxyai_http_request_duration_seconds", "Request latency", ["endpoint"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    REQUESTS.labels(request.method, request.url.path).inc()
    resp = await call_next(request)
    LATENCY.labels(request.url.path).observe(time.time() - start)
    return resp

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    upload_id = file.filename  # adjust as needed
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{upload_id}.wav")
    with open(dest, "wb") as f:
        f.write(await file.read())
    # enqueue both tasks
    transcribe_segments.delay(upload_id)
    diarize_full.delay(upload_id)
    log.info("Upload accepted", upload_id=upload_id)
    return {"upload_id": upload_id}