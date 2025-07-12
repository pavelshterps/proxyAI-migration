import time
import uuid
import json
from pathlib import Path
from contextlib import asynccontextmanager

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    Header, Depends, Request, Response, Body
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
from tasks import download_audio, preview_transcribe, diarize_full

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
    allow_origins=[*settings.ALLOWED_ORIGINS_LIST],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST
)

# Ensure necessary directories exist
for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)

# Redis client
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

# === Единый прогресс-эндпоинт ===
@app.get("/status/{upload_id}", summary="Check processing status")
async def get_status(upload_id: str, current_user=Depends(get_current_user)):
    base = Path(settings.RESULTS_FOLDER) / upload_id
    # если уже всё сделано на диске — возвращаем финальный статус
    if (base / "transcript.json").exists() and (base / "diarization.json").exists():
        return {
            "status": "diarization_done",
            "preview": json.loads(await redis.get(f"preview_result:{upload_id}") or "null"),
            "chunks_total": None,
            "chunks_done": None,
            "diarize_requested": True
        }

    raw = await redis.get(f"progress:{upload_id}")
    if raw:
        return json.loads(raw)
    # если нет прогресса — загрузка
    return {
        "status": "processing",
        "preview": None,
        "chunks_total": 0,
        "chunks_done": 0,
        "diarize_requested": False
    }

# === Unified results endpoint ===
@app.get("/results/{upload_id}", summary="Get preview, transcript or diarization")
async def get_results(upload_id: str, current_user=Depends(get_current_user), db=Depends(get_db)):
    # 1) Preview
    preview_data = await redis.get(f"preview_result:{upload_id}")
    if preview_data:
        payload = json.loads(preview_data)
        return JSONResponse(content={"results": payload["timestamps"]})

    # 2) Full transcript
    tp = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    if tp.exists():
        data = json.loads(tp.read_text(encoding="utf-8"))
        return JSONResponse(content={"results": data})

    # 3) Diarization + apply label_mapping
    dp = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    if dp.exists():
        segments = json.loads(dp.read_text(encoding="utf-8"))
        try:
            rec = await get_upload_for_user(db, current_user.id, upload_id)
            mapping = rec.label_mapping or {}
        except:
            mapping = {}
        for seg in segments:
            seg["speaker"] = mapping.get(str(seg["speaker"]), seg["speaker"])
        return JSONResponse(content={"results": segments})

    raise HTTPException(404, "Results not ready")

# === Загрузка файлов ===
@app.post("/upload/", dependencies=[Depends(get_current_user)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    cid = x_correlation_id or str(uuid.uuid4().hex)
    data = await file.read()
    if not data:
        raise HTTPException(400, "File is empty")
    ext = Path(file.filename).suffix or ""
    upload_id = uuid.uuid4().hex
    dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
    dest.write_bytes(data)

    try:
        await create_upload_record(db, current_user.id, upload_id)
    except Exception as e:
        log.warning("Failed to create upload record", error=str(e))

    log.bind(correlation_id=cid, upload_id=upload_id, user_id=current_user.id).info("upload accepted")
    await redis.set(f"external:{upload_id}", upload_id)

    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)

    # инициализируем «0%»
    init = {
        "status": "processing",
        "preview": None,
        "chunks_total": 0,
        "chunks_done": 0,
        "diarize_requested": False
    }
    await redis.set(f"progress:{upload_id}", json.dumps(init, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(init, ensure_ascii=False))

    return JSONResponse(
        {"upload_id": upload_id, "external_id": upload_id},
        headers={"X-Correlation-ID": cid}
    )

# === Запрос диаризации ===
@app.post("/diarize/{upload_id}", summary="Request diarization")
async def request_diarization(upload_id: str, current_user=Depends(get_current_user)):
    await redis.set(f"diarize_requested:{upload_id}", "1")
    diarize_full.delay(upload_id, None)
    return JSONResponse({"message": "diarization started"})

# === Сохранение меток спикеров ===
@app.post("/labels/{upload_id}", summary="Save speaker labels")
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    try:
        rec = await get_upload_for_user(db, current_user.id, upload_id)
    except:
        raise HTTPException(404, "upload_id not found")

    rec.label_mapping = mapping
    await db.commit()

    out_file = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    updated = []
    if out_file.exists():
        segments = json.loads(out_file.read_text(encoding="utf-8"))
        updated = [
            {"start": seg["start"], "end": seg["end"],
             "speaker": mapping.get(str(seg["speaker"]), seg["speaker"])}
            for seg in segments
        ]
        out_file.write_text(json.dumps(updated, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    return JSONResponse({"message": "labels saved", "results": updated})

# === Роуты и статичные ресурсы ===
app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")