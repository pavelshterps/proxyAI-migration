from fastapi import FastAPI, UploadFile, HTTPException
from uuid import uuid4
import shutil, os
from celery_app import celery_app
from config.settings import settings

app = FastAPI()

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid audio format")
    uid = str(uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    # отправляем first-stage задачу с task_id=uid
    celery_app.send_task(
        "tasks.diarize_full",
        args=(dest,),
        task_id=uid,
        queue="preprocess_cpu"
    )
    return {"job_id": uid}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    async_res = celery_app.AsyncResult(job_id)
    if async_res.state in ("PENDING", "RECEIVED"):
        return {"status": async_res.state}
    if async_res.state == "SUCCESS":
        return {"status": "SUCCESS", "segments": async_res.result}
    # FAIL / RETRY / ERROR
    return {"status": async_res.state, "detail": str(async_res.result)}