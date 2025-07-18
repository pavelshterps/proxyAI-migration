import time
import uuid
import json
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI,
    UploadFile, File, Form, HTTPException,
    Header, Depends, Request, Query
)
from fastapi.responses import (
    JSONResponse, HTMLResponse, PlainTextResponse, FileResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sse_starlette.sse import EventSourceResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config.settings import settings
from database import init_models, engine, get_db
from crud import create_upload_record, get_upload_for_user
from dependencies import get_current_user

from tasks import convert_to_wav_and_preview, diarize_full
from routes import router as api_router
from admin_routes import router as admin_router

app = FastAPI(title="proxyAI", version=settings.APP_VERSION)

# rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse({"detail":"Too Many Requests"},status_code=429))

# CORS & TrustedHost
app.add_middleware(CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1","localhost"]+settings.ALLOWED_ORIGINS_LIST,
)

app.include_router(api_router)
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")

# structured logging
structlog.configure(processors=[
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(x_api_key: str = Depends(api_key_header), api_key: str = Query(None)):
    key = x_api_key or api_key
    if not key:
        raise HTTPException(401, "Missing API Key")
    return key

@app.on_event("startup")
async def on_startup():
    for attempt in range(5):
        try:
            await init_models(engine)
            log.info("DB connected", attempt=attempt)
            break
        except OSError as e:
            log.warning("DB init failed", attempt=attempt, error=str(e))
            await asyncio.sleep(2)
    for d in (settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER, settings.DIARIZER_CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
        log.debug("Ensured dir", path=str(d))

@app.get("/health", tags=["default"])
async def health(): return {"status":"ok"}
@app.get("/ready", tags=["default"])
async def ready():  return {"status":"ready"}
@app.get("/metrics", tags=["default"])
async def metrics(): return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/events/{upload_id}", tags=["default"])
async def progress_events(upload_id: str, request: Request, api_key: str = Depends(get_api_key)):
    async def event_generator():
        sub = redis.pubsub()
        await sub.subscribe(f"progress:{upload_id}")
        last_hb = time.time()
        try:
            while True:
                if await request.is_disconnected(): break
                msg = await sub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                now = time.time