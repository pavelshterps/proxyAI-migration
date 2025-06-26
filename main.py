from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from uuid import uuid4
import shutil
import os

from celery_app import celery_app  # ensure celery_app exports a Celery() instance named "celery_app"

app = FastAPI()

# 1) Mount static files under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2) Serve index.html at root
@app.get("/", response_class=FileResponse)
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(404, "Index not found")
    return index_path

# 3) API endpoints

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(400, "Only WAV uploads are supported")
    uid = str(uuid4())
    upload_folder = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
    dest = os.path.join(upload_folder, f"{uid}.wav")
    os.makedirs(upload_folder, exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Запускаем диаризацию на CPU
    celery_app.send_task("tasks.diarize_full", args=(dest,), task_id=uid)

    # Сразу планируем транскрипцию на GPU
    celery_app.send_task(
        "tasks.transcribe_segments",
        args=(dest,),
        queue="preprocess_gpu"
    )

    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}