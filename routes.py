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
async def get_results(upload_id: str):
    transcript = _read_json(Path(settings.RESULTS_FOLDER)/upload_id/"transcript.json")
    diarization = _read_json(Path(settings.RESULTS_FOLDER)/upload_id/"diarization.json")
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
    base = Path(settings.RESULTS_FOLDER) / upload_id
    d_file = base / "diarization.json"
    diarization = _read_json(d_file)
    for d in diarization:
        if d["speaker"] in mapping:
            d["speaker"] = mapping[d["speaker"]]
    d_file.write_text(
        json.dumps(diarization, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return {"detail": "Labels updated"}