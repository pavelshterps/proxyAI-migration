from fastapi import FastAPI, UploadFile, HTTPException
from uuid import uuid4
import os, shutil
from celery_app import celery_app as app_celery
from config.settings import settings

api = FastAPI()

@api.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type != "audio/wav":
        raise HTTPException(400, "Only .wav files are supported")
    uid = str(uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    app_celery.send_task("tasks.diarize_full", args=(dest,), task_id=uid)
    return {"job_id": uid}

@api.get("/result/{job_id}")
async def get_result(job_id: str):
    res = app_celery.backend.get(job_id)
    if res is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": res}