from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from uuid import uuid4

from config.settings import settings
from celery_app import celery_app

router = APIRouter()

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    upload_id = str(uuid4())
    p = Path(settings.upload_folder) / upload_id
    p.write_bytes(data)

    celery_app.send_task(
        "tasks.transcribe_segments",
        args=[upload_id],
        kwargs={"correlation_id": None}
    )
    celery_app.send_task(
        "tasks.diarize_full",
        args=[upload_id],
        kwargs={"correlation_id": None}
    )

    return {"upload_id": upload_id}

@router.get("/results/{upload_id}")
async def get_results(upload_id: str):
    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")
    return {
        "transcript": (base / "transcript.json").read_text(encoding="utf-8") if (base / "transcript.json").exists() else None,
        "diarization": (base / "diarization.json").read_text(encoding="utf-8") if (base / "diarization.json").exists() else None,
    }