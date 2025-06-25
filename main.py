import os
import shutil
from uuid import uuid4
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from celery_app import celery_app
from config.settings import settings

app = FastAPI()

# если нужен фронтенд в папке `static/`
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    uid = str(uuid4())
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    celery_app.send_task("tasks.diarize_full", args=(dest,))
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    data = celery_app.backend.get(job_id)
    if data is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": data}