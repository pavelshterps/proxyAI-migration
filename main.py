import os
import shutil
from uuid import uuid4

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from celery_app import app as celery_app
from config.settings import UPLOAD_FOLDER, ALLOWED_ORIGINS

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    # генерируем UUID и сохраняем файл во временную папку
    uid = str(uuid4())
    dest_dir = os.path.abspath(UPLOAD_FOLDER)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{uid}.wav")

    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(500, f"Failed to save upload: {e}")

    # сразу ставим задачу диаризации
    celery_app.send_task("tasks.diarize_full", args=(dest,), kwargs={})
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    # результат хранится в бэкенде Celery
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    # отдаём список сегментов
    return {"status": "SUCCESS", "segments": result}