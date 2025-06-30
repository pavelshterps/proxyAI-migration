import time
import structlog
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config.settings import settings
from tasks import transcribe_segments, diarize_full
from routes import router as api_router

# Настройка structlog для вывода JSON
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()

# Гарантируем существование директорий
Path(settings.upload_folder).mkdir(parents=True, exist_ok=True)
Path(settings.results_folder).mkdir(parents=True, exist_ok=True)
Path(settings.diarizer_cache_dir).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="proxyAI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP-метрики
REQUESTS = Counter("proxyai_http_requests_total", "HTTP requests", ["method", "endpoint"])
LATENCY = Histogram("proxyai_http_request_duration_seconds", "Request latency", ["endpoint"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    REQUESTS.labels(request.method, request.url.path).inc()
    response = await call_next(request)
    LATENCY.labels(request.url.path).observe(time.time() - start)
    return response

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    # Валидация типа
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    # Валидация размера
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")
    # Сохраняем
    upload_id = file.filename
    dest = Path(settings.upload_folder) / upload_id
    with open(dest, "wb") as f:
        f.write(data)
    # Запускаем фоновые задачи
    transcribe_segments.delay(upload_id)
    diarize_full.delay(upload_id)
    log.info("upload_accepted", upload_id=upload_id, size=len(data))
    return {"upload_id": upload_id}

# Дополнительные маршруты (транскрипция по HTTP и выдача результатов)
app.include_router(api_router)

# Статика (фронтенд)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")