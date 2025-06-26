# main.py
import os
import shutil
from uuid import uuid4

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config.settings import settings
from celery_app import celery_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve your React/HTML from ./static (must exist)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(400, "only WAV files accepted")
    uid = uuid4().hex
    out = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        shutil.copyfileobj(file.file, f)
    celery_app.send_task("tasks.diarize_full", args=(out,))
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    res = celery_app.backend.get(job_id)
    if res is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": res}