import time
import uuid
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import structlog  # логгер сразу
import redis.asyncio as redis_async
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Request, Response, Body, Query
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
from tasks import download_audio, preview_transcribe, transcribe_segments, diarize_full

# === Логгер ===
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# === Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    for attempt in range(1, 6):
        try:
            await init_models(engine)
            log.info("Database models initialized", attempt=attempt)
            break
        except OSError as e:
            log.warning("init_models failed, retrying", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    else:
        log.error("init_models permanently failed after 5 attempts")
    yield

# === FastAPI ===
app = FastAPI(title="proxyAI", version=settings.APP_VERSION, lifespan=lifespan)

# === Middleware ===
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware,
                   allow_origins=[*settings.ALLOWED_ORIGINS_LIST],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware,
                   allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST)

# === Директории ===
for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)
    log.debug("Ensured directory exists", path=str(d))

# === Redis & API Key ===
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(x_api_key: str = Depends(api_key_header), api_key: str = Query(None)):
    key = x_api_key or api_key
    if not key:
        log.warning("Missing API key")
        raise HTTPException(401, "Missing API Key")
    return key

# === Metrics ===
HTTP_REQ_COUNT   = Counter("http_requests_total", "Total HTTP requests", ["method","path"])
HTTP_REQ_LATENCY = Histogram("http_request_duration_seconds","HTTP latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return resp

# === Endpoints ===

@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    return {"status":"ok","version":app.version}

@app.get("/ready")
async def ready():
    return {"status":"ready","version":app.version}

@app.get("/metrics")
@limiter.limit("10/minute")
async def metrics_endpoint(request: Request):
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.post("/upload", dependencies=[Depends(get_current_user)])
@app.post("/upload/", dependencies=[Depends(get_current_user)])
@limiter.limit("10/minute")
async def upload(request: Request,
                 file: UploadFile = File(...),
                 x_correlation_id: str | None = Header(None),
                 current_user=Depends(get_current_user),
                 db=Depends(get_db)):
    cid = x_correlation_id or uuid.uuid4().hex
    data = await file.read()
    if not data:
        raise HTTPException(400, "File is empty")
    ext = Path(file.filename).suffix
    upload_id = uuid.uuid4().hex
    dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
    dest.write_bytes(data)
    try:
        await create_upload_record(db, current_user.id, upload_id)
    except Exception:
        pass
    await redis.set(f"external:{upload_id}", upload_id)
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)
    state = {"status":"started","preview":None}
    await redis.set(f"progress:{upload_id}", json.dumps(state))
    await redis.publish(f"progress:{upload_id}", json.dumps(state))
    return JSONResponse({"upload_id":upload_id,"external_id":upload_id}, headers={"X-Correlation-ID":cid})


@app.get("/events/{upload_id}")
async def progress_events(upload_id: str,
                          request: Request,
                          api_key: str = Depends(get_api_key)):
    async def gen():
        pub = redis.pubsub()
        await pub.subscribe(f"progress:{upload_id}")
        last = time.time()
        while True:
            if await request.is_disconnected():
                break
            msg = await pub.get_message(ignore_subscribe_messages=True, timeout=0.5)
            now = time.time()
            if msg and msg["type"]=="message":
                yield f"data: {msg['data']}\n\n"
            elif now - last > 1:
                yield ":\n\n"
                last = now
            await asyncio.sleep(0.1)
        await pub.unsubscribe(f"progress:{upload_id}")
    return EventSourceResponse(gen())


@app.get("/results/{upload_id}")
async def get_results(upload_id: str, current_user=Depends(get_current_user), db=Depends(get_db)):
    base = Path(settings.RESULTS_FOLDER) / upload_id
    p = base / "preview.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        return JSONResponse({"results": j["timestamps"], "text": j["text"]})
    t = base / "transcript.json"
    if not t.exists():
        raise HTTPException(404, "Results not ready")
    transcript = json.loads(t.read_text(encoding="utf-8"))
    d = base / "diarization.json"
    if d.exists():
        diar = json.loads(d.read_text(encoding="utf-8"))
        rec = await get_upload_for_user(db, current_user.id, upload_id)
        lm = rec.label_mapping or {}
        # apply labels in-place
        for seg in diar:
            seg["speaker"] = lm.get(str(seg["speaker"]), seg["speaker"])
        merged = []
        i = 0  # pointer в diar
        for seg in transcript:
            # оптимизация: не ищем заново каждый раз, двигаем i вперёд
            while i < len(diar)-1 and diar[i]["end"] <= seg["start"]:
                i += 1
            spk = diar[i]["speaker"] if diar[i]["start"] <= seg["start"] < diar[i]["end"] else None
            merged.append({**seg, "speaker": spk})
        return JSONResponse({"results": merged})
    return JSONResponse({"results": transcript})


@app.post("/diarize/{upload_id}")
async def request_diarization(upload_id: str, current_user=Depends(get_current_user)):
    await redis.set(f"diarize_requested:{upload_id}", "1")
    await redis.publish(f"progress:{upload_id}", json.dumps({"diarize_requested": True}))
    diarize_full.delay(upload_id, None)
    return JSONResponse({"message":"diarization started"})


@app.post("/labels/{upload_id}")
async def save_labels(upload_id: str,
                      mapping: dict = Body(...),
                      current_user=Depends(get_current_user),
                      db=Depends(get_db)):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    rec.label_mapping = mapping
    await db.commit()

    base = Path(settings.RESULTS_FOLDER) / upload_id
    # сразу пересобираем merged и возвращаем клиенту:
    transcript = json.loads((base/"transcript.json").read_text(encoding="utf-8"))
    diar = json.loads((base/"diarization.json").read_text(encoding="utf-8"))
    # применяем в памяти
    merged = []
    i = 0
    for seg in transcript:
        while i < len(diar)-1 and diar[i]["end"] <= seg["start"]:
            i += 1
        spk_raw = diar[i]["speaker"]
        spk = mapping.get(str(spk_raw), spk_raw)
        merged.append({**seg, "speaker": spk})
    return JSONResponse({"results": merged})


# === Routers & Static ===
app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")