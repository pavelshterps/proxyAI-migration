import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4

from config.settings import settings
from celery_app import celery_app

router = APIRouter()


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    upload_id = str(uuid4())
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    upload_path = os.path.join(settings.UPLOAD_FOLDER, f"{upload_id}.wav")

    # save
    with open(upload_path, "wb") as fp:
        fp.write(await file.read())

    # enqueue both tasks
    celery_app.send_task("tasks.transcribe_segments", args=[upload_id])
    celery_app.send_task("tasks.diarize_full",      args=[upload_id])

    return {"upload_id": upload_id}


@router.get("/results/{upload_id}")
def get_results(upload_id: str):
    base = os.path.join(settings.RESULTS_FOLDER, upload_id)
    transcript = os.path.join(base, "transcript.json")
    diarize    = os.path.join(base, "diarization.json")

    if not os.path.isdir(base):
        raise HTTPException(404, "upload_id not found")

    return {
        "transcript": os.path.exists(transcript) and open(transcript, "r", encoding="utf-8").read() or None,
        "diarization": os.path.exists(diarize) and open(diarize, "r", encoding="utf-8").read() or None,
    }