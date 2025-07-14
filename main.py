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
    Header, Depends, Request, Response, Body, Query
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security.api_key import APIKeyHeader
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    for attempt in range(1, 6):
        try:
            await init_models(engine)
            log.info("DB initialized", attempt=attempt)
            break
        except OSError as e:
            log.warning("DB init failed, retry", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    yield

app = FastAPI(title="proxyAI", version=settings.APP_VERSION, lifespan=lifespan)
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1","localhost"] + settings.ALLOWED_ORIGINS_LIST
)

for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)
    log.debug("Ensured dir exists", path=str(d))

redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(x_api_key: str = Depends(api_key_header), api_key: str = Query(None)):
    key = x_api_key or api_key
    if not key:
        log.warning("Missing API key")
        raise HTTPException(401, "Missing API Key")
    return key

@app.get("/events/{upload_id}", include_in_schema=False)
@app.get("/events/{upload_id}/", include_in_schema=False)
async def progress_events(upload_id: str, request: Request, api_key: str = Depends(get_api_key)):
    log.info("SSE subscribe", upload_id=upload_id)
    async def generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"progress:{upload_id}")
        last_hb = time.time()
        try:
            while True:
                if await request.is_disconnected():
                    log.info("SSE unsub before complete", upload_id=upload_id)
                    break
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                now = time.time()
                if msg and msg["type"] == "message":
                    payload = msg["data"]
                    log.debug("SSE ▶", upload_id=upload_id, data=payload)
                    yield f"data: {payload}\n\n"
                elif now - last_hb > 1.0:
                    yield ":\n\n"
                    last_hb = now
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(f"progress:{upload_id}")
            log.info("SSE unsubscribed", upload_id=upload_id)
    return EventSourceResponse(generator())

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

    ext = Path(file.filename).suffix or ""
    upload_id = uuid.uuid4().hex
    dest = Path(settings.UPLOAD_FOLDER) / f"{upload_id}{ext}"
    dest.write_bytes(data)
    log.info("File saved", upload_id=upload_id, path=str(dest), size=len(data))

    try:
        await create_upload_record(db, current_user.id, upload_id)
    except Exception:
        log.warning("DB record create failed", upload_id=upload_id)

    await redis.set(f"external:{upload_id}", upload_id)
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)

    init_state = {"status":"started","preview":None}
    await redis.set(f"progress:{upload_id}", json.dumps(init_state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(init_state, ensure_ascii=False))

    return JSONResponse({"upload_id":upload_id,"external_id":upload_id},
                        headers={"X-Correlation-ID":cid})

@app.get("/results/{upload_id}", summary="Get transcript + speakers")
async def get_results(upload_id: str, current_user=Depends(get_current_user), db=Depends(get_db)):
    base = Path(settings.RESULTS_FOLDER) / upload_id

    tp = base / "transcript.json"
    if tp.exists():
        transcript = json.loads(tp.read_text(encoding="utf-8"))
        dp = base / "diarization.json"
        if dp.exists():
            raw = json.loads(dp.read_text(encoding="utf-8"))
            rec = await get_upload_for_user(db, current_user.id, upload_id)
            lm  = rec.label_mapping or {}
            merged = []
            for seg in transcript:
                orig = next((d["speaker"] for d in raw if d["start"] <= seg["start"] < d["end"]), None)
                speaker = lm.get(str(orig), orig)
                merged.append({
                    "start": seg["start"], "end": seg["end"], "text": seg["text"],
                    "orig": orig, "speaker": speaker
                })
            return JSONResponse({"results": merged})
        return JSONResponse({"results": transcript})

    pf = base / "preview.json"
    if pf.exists():
        pl = json.loads(pf.read_text(encoding="utf-8"))
        return JSONResponse({"results": pl["timestamps"], "text": pl["text"]})

    raise HTTPException(404, "Results not ready")

@app.post("/diarize/{upload_id}", summary="Request diarization")
async def request_diarization(upload_id: str, current_user=Depends(get_current_user)):
    await redis.set(f"diarize_requested:{upload_id}", "1")
    state = json.loads(await redis.get(f"progress:{upload_id}") or "{}")
    state["diarize_requested"] = True
    await redis.set(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    await redis.publish(f"progress:{upload_id}", json.dumps(state, ensure_ascii=False))
    diarize_full.delay(upload_id, None)
    return JSONResponse({"message":"diarization started"})

@app.post("/labels/{upload_id}", summary="Save speaker labels")
async def save_labels(upload_id: str, mapping: dict = Body(...),
                      current_user=Depends(get_current_user), db=Depends(get_db)):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    rec.label_mapping = mapping
    await db.commit()

    base = Path(settings.RESULTS_FOLDER) / upload_id
    raw = json.loads((base / "diarization.json").read_text(encoding="utf-8"))
    updated = []
    for seg in raw:
        new_spk = mapping.get(str(seg["speaker"]), seg["speaker"])
        updated.append({"start":seg["start"], "end":seg["end"], "speaker":new_spk})
    (base / "diarization.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2))

    transcript = json.loads((base / "transcript.json").read_text(encoding="utf-8"))
    merged = []
    for seg in transcript:
        orig = next((d["speaker"] for d in raw if d["start"] <= seg["start"] < d["end"]), None)
        speaker = mapping.get(str(orig), orig)
        merged.append({
            "start": seg["start"], "end": seg["end"], "text": seg["text"],
            "orig": orig, "speaker": speaker
        })
    return JSONResponse({"results": merged})

app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")