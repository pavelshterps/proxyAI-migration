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

from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from database import get_db, engine, init_models
from crud import create_upload_record, get_upload_for_user
from dependencies import get_current_user
from routes import router as api_router
from admin_routes import router as admin_router
from tasks import download_audio, preview_transcribe, diarize_full  # убрали transcribe_segments

# === FastAPI lifecycle ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Retry init_models при старте сервера."""
    for _ in range(5):
        try:
            await init_models(engine)
            break
        except OSError as e:
            log.warning("init_models failed, retrying", error=str(e))
            await asyncio.sleep(2)
    else:
        log.error("init_models permanently failed")
    yield

app = FastAPI(title="proxyAI", version=settings.APP_VERSION, lifespan=lifespan)

# === Logging & Rate limiting ===

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# === CORS & Trusted Hosts ===

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

# === Ensure directories exist ===

for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)

# === Redis & Security ===

redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# === Prometheus metrics ===

HTTP_REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return response

# === Health and readiness ===

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

# === Core API routes ===

@app.post("/upload/", dependencies=[Depends(get_current_user)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    """
    При загрузке:
      1) Сохраняем файл
      2) Создаём запись в БД
      3) Инициируем download_audio и preview_transcribe —
         транскрипция full будет запущена из preview_transcribe
      4) Инициализируем прогресс в Redis
    """
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
    except Exception:
        log.warning("Failed to create upload record", upload_id=upload_id)

    log.bind(correlation_id=cid, upload_id=upload_id, user_id=current_user.id).info("upload accepted")
    await redis.set(f"external:{upload_id}", upload_id)

    # Стартуем цепочку обработки
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)
    # убрали: transcribe_segments.delay(upload_id, cid)

    # Инициализируем прогресс
    state = {"status": "started", "preview": None}
    await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    return JSONResponse(
        {"upload_id": upload_id, "external_id": upload_id},
        headers={"X-Correlation-ID": cid}
    )

@app.get("/events/{upload_id}")
async def progress_events(upload_id: str):
    """
    SSE stream прогресса обработки из Redis.
    """
    async def generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"progress:{upload_id}")
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg["type"] == "message":
                    yield f"data: {msg['data']}\n\n"
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(f"progress:{upload_id}")

    return EventSourceResponse(generator())

@app.get("/status/{upload_id}", summary="Check processing status")
async def get_status(upload_id: str, current_user=Depends(get_current_user)):
    raw = await redis.get(f"progress:{upload_id}") or "{}"
    return json.loads(raw)

@app.get("/results/{upload_id}", summary="Get preview, transcript or diarization")
async def get_results(upload_id: str, current_user=Depends(get_current_user), db=Depends(get_db)):
    """
    Возвращаем:
      1) Preview, если есть
      2) Полный transcript.json
      3) diarization.json (с учётом label_mapping)
    """
    # 1) preview
    pd = await redis.get(f"preview_result:{upload_id}")
    if pd:
        pl = json.loads(pd)
        return JSONResponse(content={"results": pl["timestamps"], "text": pl["text"]})

    # 2) полный транскрипт
    tp = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    if tp.exists():
        data = json.loads(tp.read_text(encoding="utf-8"))
        return JSONResponse(content={"results": data})

    # 3) diarization + mapping
    dp = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    if dp.exists():
        segs = json.loads(dp.read_text(encoding="utf-8"))
        rec = await get_upload_for_user(db, current_user.id, upload_id)
        mapping = rec.label_mapping or {}
        for s in segs:
            s["speaker"] = mapping.get(str(s["speaker"]), s["speaker"])
        return JSONResponse(content={"results": segs})

    raise HTTPException(404, "Results not ready")

@app.post("/diarize/{upload_id}", summary="Request diarization")
async def request_diarization(upload_id: str, current_user=Depends(get_current_user)):
    """
    Поздний запрос диаризации: ставим флаг и запускаем diarize_full.
    """
    await redis.set(f"diarize_requested:{upload_id}", "1")
    state = json.loads(await redis.get(f"progress:{upload_id}") or "{}")
    state["diarize_requested"] = True
    await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))

    diarize_full.delay(upload_id, None)
    return JSONResponse({"message": "diarization started"})

@app.post("/labels/{upload_id}", summary="Save speaker labels")
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Сохраняем соответствие спикеров и обновляем diarization.json на диске.
    """
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    rec.label_mapping = mapping
    await db.commit()

    out = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    updated = []
    if out.exists():
        segs = json.loads(out.read_text(encoding="utf-8"))
        updated = [
            {"start": s["start"], "end": s["end"],
             "speaker": mapping.get(str(s["speaker"]), s["speaker"])}
            for s in segs
        ]
        out.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    return JSONResponse({"results": updated})

# === Подключаем остальные маршруты и статику ===

app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")