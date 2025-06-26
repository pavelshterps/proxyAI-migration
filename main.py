from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from uuid import uuid4
import os, shutil
from celery_app import celery_app
from config.settings import settings

app = FastAPI()

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type != "audio/wav":
        raise HTTPException(400, "Only WAV is supported")
    uid = str(uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    # сначала диаризация, затем цепочка транскрибирования чанков
    celery_app.send_task("tasks.diarize_full", args=(dest,))
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}

@app.get("/")
async def root():
    index = os.path.join("static", "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"detail": "Not Found"}