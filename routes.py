import time
import json
from pathlib import Path
from typing import Dict

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
    Если для этого upload_id ранее были сохранены метки в labels.json, они применяются.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id

    transcript_path = base / "transcript.json"
    diarization_path = base / "diarization.json"
    labels_path = base / "labels.json"

    if not transcript_path.exists() or not diarization_path.exists():
        raise HTTPException(status_code=404, detail="Results not ready")

    # Загрузка исходных данных
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))

    # Загрузка сохранённых меток (если есть)
    if labels_path.exists():
        labels_map = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        labels_map = {}

    enriched = []
    for seg in transcript:
        # ищем исходного спикера по таймкоду
        orig_spk = next(
            (d["speaker"] for d in diarization
             if d["start"] <= seg["start"] < d["end"]),
            None
        ) or "unknown"

        # применяем переименование, если оно задано
        final_spk = labels_map.get(orig_spk, orig_spk)

        enriched.append({
            "segment": seg["segment"],
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
            "speaker": final_spk,
            "time":    time.strftime("%H:%M:%S", time.gmtime(seg["start"]))
        })

    return {"results": enriched}


@router.post("/labels/{upload_id}")
async def update_labels(
    upload_id: str,
    labels: Dict[str, str],  # ожидаем JSON вида { "spk_0": "Alice", "spk_1": "Bob", ... }
    current_user=Depends(get_current_user),
):
    """
    Сохраняет/обновляет маппинг спикеров в labels.json,
    чтобы при следующем запросе к /results переименования применялись автоматически.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    diar_path = base / "diarization.json"
    labels_path = base / "labels.json"

    if not diar_path.exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    # Очищаем пустые метки и сохраняем
    cleaned = {orig: new.strip() for orig, new in labels.items() if new.strip()}
    labels_path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {"updated": True, "labels": cleaned}