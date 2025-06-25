from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
import shutil
import os
from celery_app import celery_app

# Создаём FastAPI
app = FastAPI()

# Монтируем каталог static, чтобы GET / возвращал static/index.html
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.post("/transcribe")
async def start_transcription(file: UploadFile = File(...)):
    # Проверяем, что это WAV
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav files are accepted")
    uid = str(uuid4())
    dest_folder = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
    os.makedirs(dest_folder, exist_ok=True)
    dest = os.path.join(dest_folder, f"{uid}.wav")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    # Запускаем цепочку: сначала диаризация, потом транскрипция сегментов
    celery_app.send_task("tasks.diarize_full", args=(dest,), queue="preprocess_cpu")
    return {"job_id": uid}


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    # Забираем из бэкенда celery результат по ключу job_id
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}