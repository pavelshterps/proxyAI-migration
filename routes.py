# routes.py

import time
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

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
        # находим спикера по началу сегмента
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
            # строковое время в формате ЧЧ:ММ:СС
            "time": time.strftime("%H:%M:%S", time.gmtime(seg["start"]))
        })

    return {"results": enriched}