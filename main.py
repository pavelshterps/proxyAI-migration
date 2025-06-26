import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from config.settings import settings
from celery_app import celery_app

app = FastAPI()

@app.post("/transcribe")
async def start_transcription(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    uid = str(uuid.uuid4())
    upload_dir = settings.UPLOAD_FOLDER
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, f"{uid}.wav")
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    celery_app.send_task(
        "tasks.diarize_full",
        args=[dest],
        queue="preprocess_cpu",
        task_id=uid,
    )
    return {"job_id": uid}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    async_result = celery_app.AsyncResult(job_id, backend=settings.CELERY_RESULT_BACKEND)
    if async_result.state == "PENDING":
        return {"status": "PENDING"}
    if async_result.state == "SUCCESS":
        return {"status": "SUCCESS", "segments": async_result.result}
    return {"status": async_result.state, "detail": str(async_result.info)}