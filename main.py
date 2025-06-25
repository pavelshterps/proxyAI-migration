import os
import shutil
from uuid import uuid4
from fastapi import FastAPI, UploadFile, HTTPException
from celery import chain

from config.settings import settings
from celery_app import celery_app
from tasks import diarize_full, transcribe_full

app = FastAPI()

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only WAV files accepted")
    uid = str(uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    # цепочка: сначала диаризация, потом транскрипция
    chain(
        diarize_full.s(dest),
        transcribe_full.s()
    ).apply_async(task_id=uid)
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    res = celery_app.AsyncResult(job_id)
    if res.state in ["PENDING", "STARTED", "RETRY"]:
        return {"status": "PENDING"}
    if res.successful():
        return {"status": "SUCCESS", **res.result}
    return {"status": res.state, "error": str(res.result)}