import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Response
from fastapi.websockets import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from config.settings import settings
from database import get_db
from crud import get_upload_for_user
from models import User
from fastapi.security.api_key import APIKeyHeader

router = APIRouter()

# re-use API-Key auth from main.py
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(
    key: str = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
):
    from crud import get_user_by_api_key
    if not key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    user = await get_user_by_api_key(db, key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return user

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    x_correlation_id: str | None = Header(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # identical to /upload but generates new UUID
    import uuid
    cid = x_correlation_id or str(uuid.uuid4())

    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    upload_id = str(uuid.uuid4())
    p = Path(settings.upload_folder) / upload_id
    p.write_bytes(data)

    await crud.create_upload_record(db, current_user.id, upload_id)

    from tasks import transcribe_segments, diarize_full
    transcribe_segments.delay(upload_id, correlation_id=cid)
    diarize_full.delay(upload_id, correlation_id=cid)

    return Response(
        content={"upload_id": upload_id},
        headers={"X-Correlation-ID": cid}
    )

@router.get("/results/{upload_id}")
async def get_results(
    upload_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # ownership check
    upload = await get_upload_for_user(db, current_user.id, upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="upload_id not found")

    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="Results not yet available")

    # load raw
    transcript_raw = (base / "transcript.json").read_text(encoding="utf-8") if (base / "transcript.json").exists() else "[]"
    diar_raw = (base / "diarization.json").read_text(encoding="utf-8") if (base / "diarization.json").exists() else "[]"

    # apply labels if present
    labels_path = base / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        # remap speakers in transcript
        trans_list = json.loads(transcript_raw)
        for seg in trans_list:
            seg_speaker = seg.get("speaker")
            if seg_speaker in labels:
                seg["speaker"] = labels[seg_speaker]
        transcript = json.dumps(trans_list, ensure_ascii=False)
        # remap diarization
        diar_list = json.loads(diar_raw)
        for seg in diar_list:
            sp = seg.get("speaker")
            if sp in labels:
                seg["speaker"] = labels[sp]
        diarization = json.dumps(diar_list, ensure_ascii=False)
    else:
        transcript, diarization = transcript_raw, diar_raw

    return {"transcript": transcript, "diarization": diarization}

@router.post("/labels/{upload_id}")
async def set_labels(
    upload_id: str,
    mapping: dict[str, str],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # ownership check
    upload = await get_upload_for_user(db, current_user.id, upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="upload_id not found")

    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="Results not yet available")

    labels_path = base / "labels.json"
    labels_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "labels saved", "labels": mapping}