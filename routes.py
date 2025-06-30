import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Response
from pathlib import Path

from config.settings import settings
from celery_app import celery_app
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe(
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None)
):
    cid = x_correlation_id or str(uuid.uuid4())

    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    upload_id = str(uuid.uuid4())
    p = Path(settings.upload_folder) / upload_id
    p.write_bytes(data)

    # передаём Correlation-ID
    celery_app.send_task(
        "tasks.transcribe_segments",
        args=[upload_id],
        kwargs={"correlation_id": cid}
    )
    celery_app.send_task(
        "tasks.diarize_full",
        args=[upload_id],
        kwargs={"correlation_id": cid}
    )

    return Response(
        content={"upload_id": upload_id},
        headers={"X-Correlation-ID": cid}
    )

@router.get("/results/{upload_id}")
@limiter.limit("20/minute")
async def get_results(upload_id: str):
    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")
    transcript = base / "transcript.json"
    diar = base / "diarization.json"
    return {
        "transcript": transcript.read_text(encoding="utf-8") if transcript.exists() else None,
        "diarization": diar.read_text(encoding="utf-8") if diar.exists() else None,
    }