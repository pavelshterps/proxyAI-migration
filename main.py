import time
import uuid
from pathlib import Path

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Response, Header,
    Depends, WebSocket, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db, engine
from crud import get_user_by_api_key, create_upload_record, get_upload_for_user
from models import Base
from routes import router as api_router
from admin_routes import router as admin_router

# создаём таблицы (переехать на Alembic в будущем)
Base.metadata.create_all(bind=engine.sync_engine)

# structlog
structlog.configure(
    processors=[...]
)
log = structlog.get_logger()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="proxyAI", version="13.7.8")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

redis = redis_async.from_url(settings.celery_broker_url, decode_responses=True)

# Auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def get_current_user(key: str = Depends(api_key_header), db: AsyncSession = Depends(get_db)):
    if not key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    user = await get_user_by_api_key(db, key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return user

# Ensure dirs
for d in (...):
    Path(d).mkdir(parents=True, exist_ok=True)

# Middlewares
app.add_middleware(...)
app.add_middleware(...)

# Metrics
HTTP_REQ_COUNT = Counter(...)
HTTP_REQ_LATENCY = Histogram(...)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    ...

@app.get("/health")
...
@app.get("/ready")
...
@app.get("/metrics")
...

@app.get(
    "/status/{upload_id}",
    summary="Check processing status",
    responses={401: {"description": "Invalid X-API-Key"}, 404: {"description": "Not found"}}
)
async def get_status(
    upload_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # проверяем принадлежность
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(status_code=404, detail="upload_id not found")
    base = Path(settings.results_folder) / upload_id
    done = (base / "transcript.json").exists() and (base / "diarization.json").exists()
    if done:
        status_str = "done"
    else:
        uploaded = (Path(settings.upload_folder) / upload_id).exists()
        status_str = "processing" if uploaded else "queued"
    return {"status": status_str}

@app.post("/upload/", dependencies=[Depends(get_current_user)])
...
# остаётся без изменений

app.include_router(api_router, tags=["proxyAI"], dependencies=[Depends(get_current_user)])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")