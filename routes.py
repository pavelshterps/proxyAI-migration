# routes.py

import time
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config.settings import settings
from dependencies import get_current_user

router = APIRouter()

class LabelsPayload(BaseModel):
    __root__: dict[str, str]

@router.get("/results/{upload_id}")
async def get_results(
    upload_id: str,
    current_user=Depends(get_current_user),
):
    """
    Возвращает объединённый список сегментов с текстом, временем и спикером.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    transcript_path = base / "transcript.json"
    diarization_path = base / "diarization.json"

    if not transcript_path.exists() or not diarization_path.exists():
        raise HTTPException(status_code=404, detail="Results not ready")

    transcript  = json.loads(transcript_path.read_text(encoding="utf-8"))
    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))

    enriched = []
    for seg in transcript:
        spk = next(
            (d["speaker"] for d in diarization
             if d["start"] <= seg["start"] < d["end"]),
            None
        )
        enriched.append({
            "segment": seg["segment"],
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
            "speaker": spk or "unknown",
            "time":    time.strftime("%H:%M:%S", time.gmtime(seg["start"]))
        })

    return {"results": enriched}

@router.post("/labels/{upload_id}")
async def save_labels(
    upload_id: str,
    payload: LabelsPayload,
    current_user=Depends(get_current_user),
):
    """
    Перезаписывает файл diarization.json новым маппингом спикеров.
    Ожидает JSON вида { "old_name": "new_name", ... }.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    diarization_path = base / "diarization.json"
    if not diarization_path.exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    # загрузим исходный json
    diar = json.loads(diarization_path.read_text(encoding="utf-8"))
    mapping = payload.__root__

    # применим переименование
    for turn in diar:
        old = turn["speaker"]
        if old in mapping:
            turn["speaker"] = mapping[old]

    # сохраним обратно
    diarization_path.write_text(
        json.dumps(diar, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {"detail": "Labels saved"}