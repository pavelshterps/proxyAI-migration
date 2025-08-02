import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

import redis.asyncio as redis_async

from config.settings import settings
from dependencies import get_current_user
from database import get_db
import crud
from tasks import merge_speakers, send_webhook_event

router = APIRouter(prefix="", dependencies=[Depends(get_current_user)])

redis = redis_async.from_url(settings.REDIS_URL, decode_responses=True)


def _read_json(p: Path) -> Any:
    if not p.exists():
        raise HTTPException(404, f"{p.name} not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/transcription/{upload_id}", tags=["default"])
async def raw_transcription(
    upload_id: str,
    pad: float = Query(0.0, description="padding for speaker merge; 0 => no merge"),
    include_orig: bool = Query(False, description="include orig label field"),
):
    try:
        base = Path(settings.RESULTS_FOLDER) / upload_id
        tp = base / "transcript.json"
        transcript = _read_json(tp)

        if pad > 0:
            dp = base / "diarization.json"
            if dp.exists():
                diar = _read_json(dp)
                merged = merge_speakers(transcript, diar, pad=pad)
                if not include_orig:
                    for seg in merged:
                        seg.pop("orig", None)
                return {"transcript": merged}

        return {"transcript": transcript}
    except HTTPException as e:
        return {"status": "error", "detail": e.detail}


@router.get("/transcription/{upload_id}/preview", tags=["default"])
async def raw_preview(upload_id: str):
    try:
        p = Path(settings.RESULTS_FOLDER) / upload_id / "preview_transcript.json"
        return {"preview": _read_json(p)}
    except HTTPException as e:
        return {"status": "error", "detail": e.detail}


@router.get("/diarization/{upload_id}", tags=["default"])
async def raw_diarization(
    upload_id: str,
    pad: float = Query(0.0, description="padding for merge against transcript; 0 => raw diar"),
    include_orig: bool = Query(False, description="include orig label in merged output"),
):
    base = Path(settings.RESULTS_FOLDER) / upload_id
    if not base.exists():
        return JSONResponse(status_code=404, content={"status": "not_found"})

    diar_file = base / "diarization.json"
    if not diar_file.exists():
        return JSONResponse(status_code=202, content={"status": "processing"})

    try:
        if pad > 0:
            tp = base / "transcript.json"
            if tp.exists():
                transcript = _read_json(tp)
                diar = _read_json(diar_file)
                merged = merge_speakers(transcript, diar, pad=pad)
                if not include_orig:
                    for seg in merged:
                        seg.pop("orig", None)
                return JSONResponse(status_code=200, content={"status": "done", "merged": merged})

        diar = _read_json(diar_file)
        return JSONResponse(status_code=200, content={"status": "done", "diarization": diar})
    except HTTPException as e:
        return {"status": "error", "detail": e.detail}


@router.post("/labels/{upload_id}", tags=["default"])
async def save_labels(
    upload_id: str,
    mapping: dict,
    current=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ok = await crud.update_label_mapping(db, current.id, upload_id, mapping)
    if not ok:
        raise HTTPException(404, "upload_id not found")

    base = Path(settings.RESULTS_FOLDER) / upload_id
    dfile = base / "diarization.json"
    dia = _read_json(dfile)
    for seg in dia:
        key = str(seg["speaker"])
        if key in mapping:
            seg["speaker"] = mapping[key]
    dfile.write_text(json.dumps(dia, ensure_ascii=False, indent=2), encoding="utf-8")

    # пушим обновление в SSE/redis, чтобы фронт моментально видел смену labels
    await redis.set(f"progress:{upload_id}", json.dumps({"status": "labels_updated"}))
    await redis.publish(
        f"progress:{upload_id}",
        json.dumps({"status": "labels_updated", "mapping": mapping}),
    )

    send_webhook_event("labels_updated", upload_id, {"mapping": mapping})
    return {"detail": "Labels updated"}