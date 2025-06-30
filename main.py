from fastapi import FastAPI, UploadFile, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from uuid import uuid4

import shutil
import os
import time

from celery_app import celery_app
from celery.result import AsyncResult


app = FastAPI()

# Статические файлы фронтенда
app.mount("/static", StaticFiles(directory="static"), name="static")

# Prometheus-метрики
REQUEST_COUNT = Counter(
    "proxyai_request_count",
    "Total HTTP requests",
    ["method", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "proxyai_request_latency_seconds",
    "Latency per endpoint",
    ["endpoint"]
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(elapsed)
    return response


@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)


@app.get("/", response_class=FileResponse)
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(404, "Index not found")
    return index_path


@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(400, "Only WAV uploads are supported")
    uid = str(uuid4())
    upload_folder = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
    os.makedirs(upload_folder, exist_ok=True)
    dest = os.path.join(upload_folder, f"{uid}.wav")
    with open(dest, "wb") as out_f:
        shutil.copyfileobj(file.file, out_f)
    # Сначала диаризация, затем транскрипция
    celery_app.send_task("tasks.diarize_full", args=(uid,), task_id=uid)
    return {"job_id": uid}


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    async_result = AsyncResult(job_id, app=celery_app)
    if not async_result.ready():
        return {"status": async_result.state}
    # Предполагается, что tasks.transcribe_segments возвращает список сегментов
    return {"status": async_result.state, "segments": async_result.result}