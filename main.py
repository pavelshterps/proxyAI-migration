from fastapi import FastAPI, UploadFile, HTTPException
from uuid import uuid4
import os, shutil

from celery_app import app as celery_app
from config.settings import settings

app = FastAPI()

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    # only accept .wav
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(400, "Only WAV files are supported")
    uid = str(uuid4())
    dest = os.path.join(settings.upload_folder, f"{uid}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    celery_app.send_task("tasks.diarize_full", args=(dest,), task_id=uid)
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = celery_app.AsyncResult(job_id)
    if result.state == "PENDING":
        return {"status": "PENDING"}
    elif result.state == "SUCCESS":
        return {"status": "SUCCESS", "segments": result.result}
    else:
        return {"status": result.state, "detail": str(result.info)}