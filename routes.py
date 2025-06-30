from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from celery_app import celery_app
from config.settings import settings

router = APIRouter()

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Валидация типа
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    # Валидация размера
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    upload_id = str(uuid4())
    Path(settings.upload_folder).mkdir(parents=True, exist_ok=True)
    upload_path = Path(settings.upload_folder) / upload_id
    with open(upload_path, "wb") as fp:
        fp.write(data)

    celery_app.send_task("tasks.transcribe_segments", args=[upload_id])
    celery_app.send_task("tasks.diarize_full", args=[upload_id])

    return {"upload_id": upload_id}

@router.get("/results/{upload_id}")
async def get_results(upload_id: str):
    base = Path(settings.results_folder) / upload_id
    transcript = base / "transcript.json"
    diarization = base / "diarization.json"

    if not base.is_dir():
        raise HTTPException(status_code=404, detail="upload_id not found")

    return {
        "transcript": transcript.read_text(encoding="utf-8") if transcript.exists() else None,
        "diarization": diarization.read_text(encoding="utf-8") if diarization.exists() else None,
    }