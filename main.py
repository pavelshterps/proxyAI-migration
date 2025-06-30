import time
from pathlib import Path

import structlog
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config.settings import settings
from routes import router as api_router

# Настройка structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()

# Создаём необходимые директории
for folder in (settings.upload_folder, settings.results_folder, settings.diarizer_cache_dir):
    Path(folder).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="proxyAI", version="13.7.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(elapsed)
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
    upload_id = file.filename  # или можно генерировать UUID
    dest = Path(settings.upload_folder) / upload_id
    dest.write_bytes(data)
    log = log.bind(correlation_id=x_correlation_id, upload_id=upload_id, size=len(data))
    log.info("upload accepted")
    # Отправляем на обработку
    from tasks import transcribe_segments, diarize_full
    transcribe_segments.delay(upload_id)
    diarize_full.delay(upload_id)
    return {"upload_id": upload_id}

# Подключаем остальные маршруты
app.include_router(api_router, prefix="", tags=["proxyAI"])

# Статика для фронтенда
app.mount("/static", StaticFiles(directory="static"), name="static")