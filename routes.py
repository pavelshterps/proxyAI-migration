# routes.py

import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from config.settings import settings
from dependencies import get_current_user

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
    """
    Вернёт только транскрипцию, как она лежит в transcript.json
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    t_file = base / "transcript.json"
    transcript = _read_json(t_file)
    return {"transcript": transcript}

@router.get("/diarization/{upload_id}")
async def get_diarization(upload_id: str):
    """
    Вернёт только диаризацию, как она лежит в diarization.json
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    d_file = base / "diarization.json"
    diarization = _read_json(d_file)
    return {"diarization": diarization}

@router.get("/results/{upload_id}")
async def get_results(upload_id: str):
    """
    Скомбинированный вид для UI: транскрипция + подстановка спикера
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    transcript = _read_json(base / "transcript.json")
    diarization = _read_json(base / "diarization.json")

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
async def save_labels(upload_id: str, mapping: dict, current=Depends(get_current_user)):
    """
    Перезаписать имена спикеров в diarization.json по словарю mapping.
    """
    base = Path(settings.RESULTS_FOLDER) / upload_id
    d_file = base / "diarization.json"
    diarization = _read_json(d_file)

    for d in diarization:
        if d["speaker"] in mapping:
            d["speaker"] = mapping[d["speaker"]]

    d_file.write_text(json.dumps(diarization, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"detail": "Labels updated"}