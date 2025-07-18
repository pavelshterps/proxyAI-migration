import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from dependencies import get_current_user
from database import get_db
import crud

router = APIRouter(
    prefix="",
    dependencies=[Depends(get_current_user)],
)

def _read_json(path: Path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/transcription/{upload_id}", tags=["default"])
async def get_transcription(upload_id: str):
    t = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    return {"transcript": _read_json(t)}


@router.get("/diarization/{upload_id}", tags=["default"])
async def get_diarization(upload_id: str):
    d = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    return {"diarization": _read_json(d)}


@router.get("/transcription/{upload_id}/preview", tags=["default"])
async def get_preview(upload_id: str):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "preview_transcript.json"
    return {"preview": _read_json(p)}


@router.post("/labels/{upload_id}", tags=["default"])
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    updated = await crud.update_label_mapping(db, current.id, upload_id, mapping)
    if not updated:
        raise HTTPException(404, "upload_id not found")

    # пересохраняем speakers в diarization.json
    base = Path(settings.RESULTS_FOLDER) / upload_id
    d_file = base / "diarization.json"
    diarization = _read_json(d_file)
    for segment in diarization:
        key = str(segment["speaker"])
        if key in mapping:
            segment["speaker"] = mapping[key]
    d_file.write_text(
        json.dumps(diarization, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {"detail": "Labels updated"}