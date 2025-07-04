# routes.py

import time
import json
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel  # только если понадобится (здесь не используется)

from config.settings import settings
from dependencies import get_current_user

router = APIRouter()


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

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
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
async def update_labels(
    upload_id: str,
    labels: Dict[str, str],  # ожидаем простой JSON-объект { oldSpeaker: newSpeaker, ... }
    current_user=Depends(get_current_user),
):
    """
    Применяет переименование спикеров в файле diarization.json
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    diar_path = base / "diarization.json"

    if not diar_path.exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    # Загрузить, переработать, сохранить
    diar = json.loads(diar_path.read_text(encoding="utf-8"))
    for turn in diar:
        orig = turn["speaker"]
        if orig in labels and labels[orig].strip():
            turn["speaker"] = labels[orig].strip()

    diar_path.write_text(json.dumps(diar, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"updated": True, "labels": labels}