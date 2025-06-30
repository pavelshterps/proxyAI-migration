# main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram
import structlog

from config.settings import settings
from tasks import transcribe_segments, diarize_full

structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()

# Metrics
REQ_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQ_LATENCY = Histogram("api_request_duration_seconds", "Request latency")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/upload/{upload_id}")
@REQ_LATENCY.time()
def upload(upload_id: str, file: UploadFile, background_tasks: BackgroundTasks):
    REQ_COUNT.labels(endpoint="/upload").inc()
    if file.content_type != "audio/wav":
        raise HTTPException(400, "Only WAV supported")
    dest = settings.upload_folder / f"{upload_id}.wav"
    with open(dest, "wb") as f:
        f.write(file.file.read())
    background_tasks.add_task(diarize_full.delay, upload_id)
    background_tasks.add_task(transcribe_segments.delay, upload_id)
    return {"upload_id": upload_id}

# ... other endpoints for checking results, listing, etc.