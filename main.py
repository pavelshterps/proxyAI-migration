import os, shutil
from fastapi import FastAPI, UploadFile, HTTPException
from uuid import uuid4
from celery_app import celery_app
from config.settings import settings

app = FastAPI()

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type != "audio/wav":
        raise HTTPException(400, "Only WAV files are supported")
    uid = str(uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    celery_app.send_task("tasks.diarize_full", args=(dest,), task_id=uid)
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    res = celery_app.backend.get(job_id)
    if res is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": res}