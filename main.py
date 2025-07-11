import time
import uuid
import json
from pathlib import Path
from urllib.parse import urlparse

import structlog
import redis.asyncio as redis_async
import httpx
from fastapi import (
    FastAPI, UploadFile, File, Form, Body, HTTPException,
    Header, Depends, Request, Response
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from contextlib import asynccontextmanager

from config.settings import settings
from database import get_db, engine, init_models
from crud import create_upload_record, get_upload_for_user
from tasks import download_audio, preview_transcribe, transcribe_segments, diarize_full

# … остальной существующий код настройки FastAPI, CORS, метрик …

redis = redis_async.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

@app.post("/upload/", dependencies=[Depends(get_current_user)])
async def upload(/* как было */):
    # … сохраняем файл, в БД и Redis …
    await redis.set(f"diarize_requested:{upload_id}", "0")
    download_audio.delay(upload_id, cid)
    preview_transcribe.delay(upload_id, cid)
    await redis.publish(f"progress:{upload_id}", "0%")
    await redis.set(f"progress:{upload_id}", "0%")
    return JSONResponse({"upload_id": upload_id, "external_id": external_id},
                        headers={"X-Correlation-ID": cid})

# === новый эндпоинт для запроса диаризации ===
@app.post("/tasks/{external_id}/diarize", summary="Request diarization")
async def request_diarization(
    external_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    upload_id = await redis.get(f"external:{external_id}")
    if not upload_id:
        raise HTTPException(404, "external_id not found")

    # отмечаем в Redis, что диаризацию нужно запустить
    await redis.set(f"diarize_requested:{upload_id}", "1")

    # если full transcript уже готов — запускаем сразу
    result_path = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    if result_path.exists():
        cid = str(uuid.uuid4().hex)
        diarize_full.delay(upload_id, cid)
        await redis.publish(f"progress:{upload_id}", "diarization_started")

    return JSONResponse({"detail": "diarization requested"})

# === остальные GET /tasks/... endpoints остаются без изменений ===

# регистрация роутеров и статика
app.include_router(api_router, tags=["proxyAI"])
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="static"), name="static")