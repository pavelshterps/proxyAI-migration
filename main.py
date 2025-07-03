import time
import uuid
from pathlib import Path

import structlog
import redis.asyncio as redis_async
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Response,
    Header,
    Depends,
    Request,
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
from contextlib import asynccontextmanager

from config.settings import settings
from database import get_db, engine, init_models
from crud import create_upload_record, get_upload_for_user
from routes import router as api_router
from admin_routes import router as admin_router
from dependencies import get_current_user

# structlog configuration
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# === Lifespan context for startup/shutdown ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация базы данных и моделей
    await init_models(engine)
    yield
    # Корректное закрытие Redis при shutdown
    await redis.close()

app = FastAPI(
    title="proxyAI",
    version="13.8.5.2",
    lifespan=lifespan,
)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# CORS and Trusted Hosts
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"] + settings.ALLOWED_ORIGINS_LIST
)

# Redis Pub/Sub + key/value for progress tracking
redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# Ensure directories exist
for d in (
    settings.UPLOAD_FOLDER,
    settings.RESULTS_FOLDER,
    settings.DIARIZER_CACHE_DIR,
):
    Path(d).mkdir(parents=True, exist_ok=True)

# Prometheus metrics
HTTP_REQ_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path"],
)
HTTP_REQ_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["path"],
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    HTTP_REQ_COUNT.labels(request.method, request.url.path).inc()
    HTTP_REQ_LATENCY.labels(request.url.path).observe(time.time() - start)
    return response

# Health endpoints