import json
import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from fastapi import UploadFile, File, Header, Response
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from crud import get_upload_for_user, create_upload_record
from models import User

router = APIRouter()

# reuse get_current_user from main.py
from main import get_current_user

@router.get("/results/{upload_id}")
async def get_results(
    upload_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # ownership check
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")

    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(404, "Results not yet available")

    # raw JSON
    tpath = base / "transcript.json"
    dpath = base / "diarization.json"
    transcript_list = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else []
    diar_list = json.loads(dpath.read_text(encoding="utf-8")) if dpath.exists() else []

    # apply labels if any
    labels_path = base / "labels.json"
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

    # remap speakers & add time stamps to transcript
    for seg in transcript_list:
        spk = seg.get("speaker")
        if spk in labels:
            seg["speaker"] = labels[spk]
        # add a human-readable time string if start exists
        if "start" in seg:
            seconds = int(seg["start"])
            seg["time"] = str(datetime.timedelta(seconds=seconds))
    for seg in diar_list:
        spk = seg.get("speaker")
        if spk in labels:
            seg["speaker"] = labels[spk]

    return {"transcript": transcript_list, "diarization": diar_list}

@router.post("/labels/{upload_id}")
async def set_labels(
    upload_id: str,
    mapping: dict[str, str],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(404, "upload_id not found")
    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(404, "Results not yet available")

    # валидация ключей
    transcript = json.loads((base / "transcript.json").read_text(encoding="utf-8"))
    valid_keys = {seg["speaker"] for seg in transcript}
    invalid = [k for k in mapping if k not in valid_keys]
    if invalid:
        raise HTTPException(400, f"Invalid speaker keys: {invalid}")

    # сохраняем
    (base / "labels.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "labels saved", "labels": mapping}