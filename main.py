import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from celery_app import celery_app
from config.settings import UPLOAD_FOLDER, FASTAPI_PORT, MAX_FILE_SIZE_MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        return HTMLResponse(open("index.html", encoding="utf-8").read())
    except Exception as e:
        logger.error("Error serving index.html: %s", e)
        raise HTTPException(500, str(e))

def cleanup_upload(path: str):
    try: os.remove(path)
    except: pass

@app.post("/transcribe", status_code=202)
async def start_transcription(
    background: BackgroundTasks,
    file: UploadFile = File(...)
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE_MB} MB)")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    dest = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    with open(dest, "wb") as f:
        f.write(contents)
    # schedule removal if something goes wrong
    background.add_task(cleanup_upload, dest)
    res = celery_app.send_task("tasks.transcribe_full", args=[dest])
    logger.info("Submitted transcribe_full, task_id=%s", res.id)
    return JSONResponse({"transcription_task_id": res.id})

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    async_res = AsyncResult(task_id, app=celery_app)
    state = async_res.state
    if state in ("PENDING", "STARTED", "RETRY"):
        return JSONResponse({"status": state}, status_code=202)
    if state == "SUCCESS":
        data = async_res.get()
        return JSONResponse({"status": state, **data}, status_code=200)
    logger.error("Task %s failed: %s", task_id, async_res.info)
    return JSONResponse({"status": state, "error": str(async_res.info)}, status_code=500)