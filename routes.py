# routes.py
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

@router.get("/transcription/{upload_id}", summary="Get raw transcript")
async def get_transcription(upload_id: str):
    t = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    return {"transcript": _read_json(t)}

@router.get("/transcription/{upload_id}/preview", summary="Get raw preview")
async def get_preview(upload_id: str):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "preview_transcript.json"
    return {"preview": _read_json(p)}

@router.get("/diarization/{upload_id}", summary="Get raw diarization")
async def get_diarization(upload_id: str):
    d = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    return {"diarization": _read_json(d)}

@router.post("/labels/{upload_id}", summary="Update speaker labels")
async def save_labels(
    upload_id: str,
    mapping: dict = Body(...),
    current=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # обновляем и в БД
    ok = await crud.update_label_mapping(db, current.id, upload_id, mapping)
    if not ok:
        raise HTTPException(404, "upload_id not found")

    # а тут обновляем локальный файл, чтобы UI сразу подхватил новые имена
    base = Path(settings.RESULTS_FOLDER) / upload_id
    dfile = base / "diarization.json"
    diar = _read_json(dfile)
    for seg in diar:
        key = str(seg["speaker"])
        if key in mapping:
            seg["speaker"] = mapping[key]
    dfile.write_text(
        json.dumps(diar, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return {"detail": "Labels updated"}