import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from dependencies import get_current_user
from database import get_db
import crud

# импорт хелпера вебхуков из tasks
from tasks import send_webhook_event

router = APIRouter(prefix="", dependencies=[Depends(get_current_user)])


def _read_json(p: Path):
    if not p.exists():
        raise HTTPException(404, f"{p.name} not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/transcription/{upload_id}", tags=["default"])
async def raw_transcription(upload_id: str):
    try:
        p = Path(settings.RESULTS_FOLDER)/upload_id/"transcript.json"
        return {"transcript": _read_json(p)}
    except HTTPException as e:
        return {"status":"error","detail":e.detail}


@router.get("/transcription/{upload_id}/preview", tags=["default"])
async def raw_preview(upload_id: str):
    try:
        p = Path(settings.RESULTS_FOLDER)/upload_id/"preview_transcript.json"
        return {"preview": _read_json(p)}
    except HTTPException as e:
        return {"status":"error","detail":e.detail}


@router.get("/diarization/{upload_id}", tags=["default"])
async def raw_diarization(upload_id: str):
    try:
        p = Path(settings.RESULTS_FOLDER)/upload_id/"diarization.json"
        return {"diarization": _read_json(p)}
    except HTTPException as e:
        return {"status":"error","detail":e.detail}


@router.post("/labels/{upload_id}", tags=["default"])
async def save_labels(upload_id: str, mapping: dict,
                      current=Depends(get_current_user),
                      db: AsyncSession=Depends(get_db)):
    ok = await crud.update_label_mapping(db, current.id, upload_id, mapping)
    if not ok:
        raise HTTPException(404, "upload_id not found")

    # обновляем на диске
    base = Path(settings.RESULTS_FOLDER)/upload_id
    dfile = base/"diarization.json"
    dia = _read_json(dfile)
    for seg in dia:
        key = str(seg["speaker"])
        if key in mapping:
            seg["speaker"] = mapping[key]
    dfile.write_text(json.dumps(dia, ensure_ascii=False, indent=2), encoding="utf-8")

    # webhook: метки спикеров обновлены
    send_webhook_event("labels_updated", upload_id, {"mapping": mapping})

    return {"detail": "Labels updated"}