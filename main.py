from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
import shutil, os
from celery_app import app as celery_app

app = FastAPI()

# монтируем папку со статикой, отдаём index.html по GET /
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static",
)

@app.post("/transcribe")
async def start_transcription(
    file: UploadFile = File(...),
):
    """
    Принимаем multipart/form-data с полем 'file'.
    """
    uid = str(uuid4())
    dest_dir = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{uid}.wav")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # отправляем задачу в очередь diarization -> gpu-пайплайн
    celery_app.send_task("tasks.diarize_full", args=(dest,), queue="preprocess_cpu")
    return {"job_id": uid}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Возвращаем либо PENDING, либо результат в формате:
    {"status":"SUCCESS","segments":[...]}
    """
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}