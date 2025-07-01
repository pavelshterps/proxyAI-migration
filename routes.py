import json
import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Response
from fastapi import UploadFile, File, Header
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from crud import (
    get_upload_for_user,
    create_upload_record
)
from dependencies import get_current_user

router = APIRouter()

@router.get("/results/{upload_id}")
async def get_results(
    upload_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(status_code=404, detail="upload_id not found")

    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="Results not yet available")

    # load raw JSON
    tpath = base / "transcript.json"
    dpath = base / "diarization.json"
    transcript_list = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else []
    diar_list = json.loads(dpath.read_text(encoding="utf-8")) if dpath.exists() else []

    # load labels if exist
    labels_path = base / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    # remap speakers & add time stamps to transcript
    for seg in transcript_list:
        spk = seg.get("speaker")
        if spk in labels:
            seg["speaker"] = labels[spk]
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
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    rec = await get_upload_for_user(db, current_user.id, upload_id)
    if not rec:
        raise HTTPException(status_code=404, detail="upload_id not found")

    base = Path(settings.results_folder) / upload_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="Results not yet available")

    # load original transcript to get valid speaker keys
    transcript = json.loads((base / "transcript.json").read_text(encoding="utf-8"))
    valid_keys = {seg["speaker"] for seg in transcript}

    # validate mapping keys
    invalid = [k for k in mapping if k not in valid_keys]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid speaker keys: {invalid}"
        )

    # validate mapping values (1–50 chars, no angle brackets)
    for new_name in mapping.values():
        if not isinstance(new_name, str) or not (1 <= len(new_name) <= 50):
            raise HTTPException(
                status_code=400,
                detail="Label names must be 1–50 characters long"
            )
        if "<" in new_name or ">" in new_name:
            raise HTTPException(
                status_code=400,
                detail="Label names must not contain '<' or '>'"
            )

    # save mapping
    labels_path = base / "labels.json"
    labels_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return {"status": "labels saved", "labels": mapping}