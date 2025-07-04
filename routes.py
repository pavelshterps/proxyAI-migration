# routes.py

import time
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import RootModel

from config.settings import settings
from dependencies import get_current_user

router = APIRouter()


class LabelUpdate(RootModel[dict[str, str]]):
    """
    Ожидаем в теле запроса JSON-объект вида
    { "oldSpeaker": "newSpeaker", ... }
    """
    pass


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

    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not ready")

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    # если есть файл с диаризацией — загрузим, иначе оставим пустым список
    if diarization_path.exists():
        diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    else:
        diarization = []

    enriched = []
    for seg in transcript:
        # сопоставляем сегмент с говорящим
        spk = next(
            (d["speaker"]
             for d in diarization
             if d["start"] <= seg["start"] < d["end"]),
            None
        )
        enriched.append({
            "segment": seg["segment"],
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
            "speaker": spk or "unknown",
            # строковое время ЧЧ:ММ:СС
            "time":    time.strftime("%H:%M:%S", time.gmtime(seg["start"]))
        })

    # возвращаем поле results, как ждёт фронтенд
    return {"results": enriched}


@router.post("/labels/{upload_id}")
async def save_labels(
    upload_id: str,
    payload: LabelUpdate,
    current_user=Depends(get_current_user),
):
    """
    Принимает переименования спикеров и правит diarization.json.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    diarization_path = base / "diarization.json"

    if not diarization_path.exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    # читаем текущие метки
    diar = json.loads(diarization_path.read_text(encoding="utf-8"))

    # mapping хранится в payload.root по стандарту RootModel
    mapping: dict[str, str] = payload.root

    # применяем новую раскладку имён
    for turn in diar:
        orig = turn.get("speaker")
        if orig in mapping and mapping[orig].strip():
            turn["speaker"] = mapping[orig].strip()

    # сохраняем обратно
    diarization_path.write_text(
        json.dumps(diar, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {"status": "ok", "updated": mapping}