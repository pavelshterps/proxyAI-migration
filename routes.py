# routes.py

import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
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

@router.get("/transcription/{upload_id}")
async def get_transcription(upload_id: str):
    t = Path(settings.RESULTS_FOLDER) / upload_id / "transcript.json"
    return {"transcript": _read_json(t)}

@router.get("/diarization/{upload_id}")
async def get_diarization(upload_id: str):
    d = Path(settings.RESULTS_FOLDER) / upload_id / "diarization.json"
    return {"diarization": _read_json(d)}

@router.get("/transcription/{upload_id}/preview")
async def get_preview(upload_id: str):
    p = Path(settings.RESULTS_FOLDER) / upload_id / "preview_transcript.json"
    return {"preview": _read_json(p)}

@router.get("/results/{upload_id}")
async def get_results(
    upload_id: str,
    current=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    transcript = _read_json(Path(settings.RESULTS_FOLDER)/upload_id/"transcript.json")
    diarization = _read_json(Path(settings.RESULTS_FOLDER)/upload_id/"diarization.json")

    # применяем пользовательскую маппинг-таблицу, если есть
    mapping = await crud.get_label_mapping(db, current.id, upload_id)
    if mapping:
        for seg in diarization:
            key = str(seg["speaker"])
            if key in mapping:
                seg["speaker"] = mapping[key]

    results = []
    for seg in transcript:
        spk = next(
            (d["speaker"] for d in diarization
             if d["start"] <= seg["start"] < d["end"]),
            "unknown"
        )
        h, r = divmod(seg["start"], 3600)
        m, s = divmod(r, 60)
        time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        results.append({
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
            "speaker": spk,
            "time":    time_str,
        })

    return {"results": results}

@router.post("/labels/{upload_id}")
async def save_labels(
    upload_id: str,
    mapping: dict,
    current=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # сохраняем в БД
    updated = await crud.update_label_mapping(db, current.id, upload_id, mapping)
    if not updated:
        raise HTTPException(404, "upload_id not found")

    # обновляем локальный файл diarization.json
    base = Path(settings.RESULTS_FOLDER) / upload_id
    d_file = base / "diarization.json"
    diarization = _read_json(d_file)
    for d in diarization:
        key = str(d["speaker"])
        if key in mapping:
            d["speaker"] = mapping[key]
    d_file.write_text(
        json.dumps(diarization, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {"detail": "Labels updated"}