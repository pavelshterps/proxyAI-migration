import os
import shutil
from fastapi import FastAPI, UploadFile, HTTPException
from uuid import uuid4

from celery_app import celery_app
from config.settings import settings

app = FastAPI()

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only WAV files supported")
    uid = str(uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    # enqueue CPU diarization first
    celery_app.send_task(
        "tasks.diarize_full",
        args=(dest,),
        queue="preprocess_cpu",
        task_id=uid,
    )
    return {"job_id": uid}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}