import os
import shutil
from uuid import uuid4

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from config.settings import settings
from celery_app import app as celery_app

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not Found")
    return HTMLResponse(open(path, "r").read())

@app.post("/transcribe")
async def upload(file: UploadFile):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    uid = str(uuid4())
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    # отправляем задачу на CPU-диаризацию
    celery_app.send_task("tasks.diarize_full", args=[dest], task_id=uid)
    return {"job_id": uid}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    res = celery_app.AsyncResult(job_id)
    if res.status == "PENDING":
        return {"status": "PENDING"}
    if res.status == "SUCCESS":
        return {"status": "SUCCESS", "segments": res.result}
    return {"status": res.status}