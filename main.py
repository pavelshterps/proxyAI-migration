# main.py
import os
import uuid
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from celery.result import AsyncResult

from config.settings import settings
from celery_app import app as celery_app

# runtime‐create uploads folder
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)

api = FastAPI(
    title="ProxyAI",
    description="Upload audio → speaker diarization → Whisper transcription",
)

# serve only at /static
api.mount("/static", StaticFiles(directory="static"), name="static")

@api.get("/", include_in_schema=False)
async def _root():
    # redirect root to OpenAPI docs
    return RedirectResponse(url="/docs")

@api.post("/transcribe")
async def transcribe(file: UploadFile):
    data = await file.read()
    if len(data) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    task_id = str(uuid.uuid4())
    fname = f"{task_id}.wav"
    path = os.path.join(settings.UPLOAD_FOLDER, fname)
    with open(path, "wb") as f:
        f.write(data)

    celery_app.send_task(
        "tasks.diarize_full",
        args=[path],
        queue="preprocess_cpu",
        task_id=task_id
    )
    return {"task_id": task_id}

@api.get("/result/{task_id}")
async def get_result(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if res.state in ("PENDING", "STARTED"):
        return JSONResponse(status_code=202, content={"status": res.state})
    if res.failed:
        return JSONResponse(
            status_code=500,
            content={"status": "FAILURE", "error": str(res.result)}
        )
    # successful result is a list of segments
    segments = res.result
    return {"status": "SUCCESS", "segments": segments}