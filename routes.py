# routes.py

import time
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config.settings import settings
from dependencies import get_current_user

router = APIRouter()

class LabelUpdate(BaseModel):
    __root__: dict[str, str]

@router.get("/results/{upload_id}")
async def get_results(
    upload_id: str,
    current_user=Depends(get_current_user),
):
    """
    Возвращает объединённый список сегментов с текстом, временем и (по возможности) спикером.
    Если диагрмация ещё не готова — возвращаем всё равно текст.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id

    transcript_path = base / "transcript.json"
    diarization_path = base / "diarization.json"

    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not ready")

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    # если diarization доступна — загрузим, иначе пустой список
    if diarization_path.exists():
        diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    else:
        diarization = []

    enriched = []
    for seg in transcript:
        # находим спикера по началу сегмента, если есть
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

    # возвращаем именно поле transcript, как ждёт фронтенд
    return {"transcript": enriched}


@router.post("/labels/{upload_id}")
async def save_labels(
    upload_id: str,
    payload: LabelUpdate,
    current_user=Depends(get_current_user),
):
    """
    Принимает словарь переименований { oldName: newName, ... }
    и обновляет файл diarization.json, заменяя speaker → mapped value.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    diarization_path = base / "diarization.json"

    if not diarization_path.exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    # читаем текущие метки
    diar = json.loads(diarization_path.read_text(encoding="utf-8"))
    mapping: dict[str,str] = payload.__root__

    # переименовываем
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