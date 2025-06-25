# main.py
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
import shutil, os
from celery_app import celery_app
from config.settings import settings

app = FastAPI()

# serve your single‐page app
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(400, "Only WAV files accepted")
    uid = str(uuid4())
    out_path = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    celery_app.send_task("tasks.diarize_full", args=(out_path,), task_id=uid)
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}