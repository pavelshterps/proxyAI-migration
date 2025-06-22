import os
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from celery.result import AsyncResult
from celery_app import celery_app
from config.settings import UPLOAD_FOLDER, FASTAPI_PORT, MAX_FILE_SIZE_MB

# Set up structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","task":"%(name)s","message":"%(message)s"}'
)
logger = logging.getLogger("app")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve JS/CSS/etc from the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.middleware("http")
async def add_cors_and_metrics(request: Request, call_next):
    # (You can re-add your Prometheus middleware here)
    return await call_next(request)

@app.get("/metrics")
def metrics():
    # (Your Prometheus generate_latest call here)
    return StreamingResponse(b"", media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def root():
    # Always serve the bundled static/index.html
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=500, detail=f"Index not found: {index_path}")
    return HTMLResponse(content=open(index_path, "r", encoding="utf-8").read())

@app.post("/transcribe", status_code=202)
async def start_transcription(
    background: BackgroundTasks,
    file: UploadFile = File(...)
):
    data = await file.read()
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE_MB} MB)")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    dest = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    with open(dest, "wb") as out:
        out.write(data)
    background.add_task(lambda p=dest: os.remove(p))
    job = celery_app.send_task("tasks.transcribe_full", args=[dest])
    logger.info("Submitted transcribe_full task_id=%s", job.id)
    return JSONResponse({"task_id": job.id})

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    ar = AsyncResult(task_id, app=celery_app)
    state = ar.state
    if state in ("PENDING", "STARTED", "RETRY"):
        return JSONResponse({"status": state}, 202)
    if state == "SUCCESS":
        data = ar.get()
        return JSONResponse({"status": state, "text": data["text"]})
    logger.error("task failed %s: %s", task_id, ar.info)
    return JSONResponse({"status": state, "error": str(ar.info)}, 500)