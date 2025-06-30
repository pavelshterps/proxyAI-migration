import structlog
from fastapi import FastAPI, HTTPException
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter, Histogram

from config.settings import settings
from tasks import transcribe_segments, diarize_full

# Structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "HTTP запросы", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_latency_seconds", "Латентность HTTP запросов", ["endpoint"])

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    endpoint = request.url.path
    method = request.method
    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        response = await call_next(request)
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/uploads/{upload_id}/transcribe")
async def api_transcribe(upload_id: str):
    try:
        transcribe_segments.delay(upload_id)
        return {"status": "queued", "task": "transcribe_segments", "upload_id": upload_id}
    except Exception as e:
        logger.error("Failed to queue transcription", error=str(e))
        raise HTTPException(500, detail="Failed to queue transcription")


@app.post("/uploads/{upload_id}/diarize")
async def api_diarize(upload_id: str):
    try:
        diarize_full.delay(upload_id)
        return {"status": "queued", "task": "diarize_full", "upload_id": upload_id}
    except Exception as e:
        logger.error("Failed to queue diarization", error=str(e))
        raise HTTPException(500, detail="Failed to queue diarization")