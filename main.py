from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
import shutil
import os
from celery_app import celery_app

# Create FastAPI app
app = FastAPI()

# Serve frontend static files (index.html, JS, CSS, etc.)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    uid = str(uuid4())
    dest_dir = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{uid}.wav")
    with open(dest_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    # Enqueue the diarization + transcription task
    celery_app.send_task("tasks.diarize_full", args=(dest_path,), task_id=uid)
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    # Fetch result by task_id
    result = celery_app.AsyncResult(job_id).result
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}