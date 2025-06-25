from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from uuid import uuid4
import shutil
import os

from celery_app import app as celery_app

app = FastAPI()

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    return "static/index.html"

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    uid = str(uuid4())
    dest = os.path.join(os.getenv("UPLOAD_FOLDER"), f"{uid}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    celery_app.send_task("tasks.diarize_full", args=(dest,), task_id=uid)
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    # result is a list of segments
    return {"status": "SUCCESS", "segments": result}