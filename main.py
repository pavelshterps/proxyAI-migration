import time
import uuid
import json
from pathlib import Path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import structlog
import redis.asyncio as redis_async
import httpx
from fastapi import (
    FastAPI, UploadFile, File, Form, Body, HTTPException,
    Header, Depends, Request, Response
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

from config.settings import settings
from database import get_db, engine, init_models
from crud import create_upload_record, get_upload_for_user
from dependencies import get_current_user
from routes import router as api_router
from admin_routes import router as admin_router
from tasks import (
    download_audio,
    preview_transcribe,
    transcribe_segments,
    diarize_full
)

# Инициализация БД при запуске
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_models(engine)
    except Exception as e:
        structlog.get_logger().warning(f"init_models failed: {e}")
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

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# CORS & TrustedHost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*settings.ALLOWED_ORIGINS_LIST, "https://transcriber-next.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST
)

# Создаем директории, если их нет
for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)

# Redis
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# API-Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Prometheus metrics
HTTP_REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return response

@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    return {"status": "ok", "version": app.version}

@app.get("/ready")
async def ready():
    return {"status": "ready", "version": app.version}

@app.get("/metrics")
@limiter.limit("10/minute")
async def metrics_endpoint(request: Request):
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")

# === статус по внутреннему upload_id ===
@app.get("/status/{upload_id}", summary="Check processing status")
async def get_status(
    upload_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    base = Path(settings.RESULTS_FOLDER) / upload_id
    done = (base / "transcript.json").exists() and (base / "diarization.json").exists()
    progress = await redis.get(f"progress:{upload_id}") or "0%"
    return {"status": "done" if done else "processing", "progress": progress}

# === единый endpoint для выдачи preview / transcript / diarization ===
@app.get("/results/{upload_id}", summary="Get preview, full transcript or diarization")
async def get_results(
    upload_id: str,
    current_user=Depends(get_current_user)
):
    # 1) Preview из Redis
    preview_data = await redis.get(f"preview_result:{upload_id}")
    if preview_data:
        preview = json.loads(preview_data)
        return JSONResponse(status_code=200, content={"results": preview["timestamps"]})

    # 2) Полная транскрипция из файла
    transcript_path = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    if transcript_path.exists():
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        results = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in data
        ]
        return JSONResponse(status_code=200, content={"results": results})

    # 3) Диаризация из файла
    diar_path = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    if diar_path.exists():
        data = json.loads(diar_path.read_text(encoding="utf-8"))
        return JSONResponse(status_code=200, content={"results": data})

    # 4) Результатов нет — 404
    raise HTTPException(404, "Results not ready")

# === запрос диаризации ===
@app.post("/diarize/{upload_id}", summary="Request diarization")
async def request_diarization(
    upload_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    await redis.set(f"diarize_requested:{upload_id}", "1")
    # сразу публикуем событие и прогресс
    await redis.set(f"progress:{upload_id}", "diarization_requested")
    await redis.publish(f"progress:{upload_id}", "diarization_requested")
    diarize_full.delay(upload_id, str(uuid.uuid4().hex))
    return {"status": "diarization requested"}

# === upload с external_id и callbacks ===
@app.post("/upload/", dependencies=[Depends(get_current_user)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile | None = File(None),
    s3_url: str | None = Form(None),
    id: str | None = Body(None),
    upload: str | None = Body(None),
    callbacks: list[str] | None = Body(None),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    cid = x_correlation_id or str(uuid.uuid4().hex)

    if upload is not None:
        if not id:
            raise HTTPException(400, "JSON must include 'id' and 'upload'")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(upload); resp.raise_for_status()
                data = resp.content
        except Exception as e:
            raise HTTPException(400, f"Failed to fetch '{upload}': {e}")
        ext = Path(urlparse(upload).path).suffix or ""
        upload_id, external_id = uuid.uuid4().hex, id
        dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
        dest.write_bytes(data)

    elif s3_url:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(s3_url); resp.raise_for_status()
                data = resp.content
        except Exception as e:
            raise HTTPException(400, f"Failed to fetch from s3_url: {e}")
        ext = Path(urlparse(s3_url).path).suffix or ""
        upload_id, external_id = uuid.uuid4().hex, uuid.uuid4().hex
        dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
        dest.write_bytes(data)

    elif file:
        data = await file.read()
        if not data:
            raise HTTPException(400, "File is empty")
        if len(data) > settings.MAX_FILE_SIZE:
            raise HTTPException(413, "File too large")
        ext = Path(file.filename).suffix or ""
        upload_id, external_id = uuid.uuid4().hex, uuid.uuid4().hex
        dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
        dest.write_bytes(data)

    else:
        raise HTTPException(400, "Must provide file, s3_url or JSON {'id','upload'}")

    await create_upload_record(db, current_user.id, upload_id)
    log.bind(correlation_id=cid, upload_id=upload_id, user_id=current_user.id).info("upload accepted")

    await redis.set(f"callbacks:{upload_id}", json.dumps(callbacks or []))
    await redis.set(f"external:{external_id}", upload_id)

    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)

    await redis.set(f"progress:{upload_id}", "0%")
    await redis.publish(f"progress:{upload_id}", "0%")

    return JSONResponse(
        {"upload_id": upload_id, "external_id": external_id},
        headers={"X-Correlation-ID": cid}
    )

# Подключение роутеров и статики
app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")