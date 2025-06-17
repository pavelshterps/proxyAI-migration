import os
import uuid
import json
import logging
import ffmpeg
import torch
import whisperx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response
from werkzeug.utils import secure_filename
from datetime import datetime
from celery.result import AsyncResult
from tasks import transcribe_task, cleanup_files
from tasks import get_file_path_by_task_id
from config.settings import settings

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_models():
    global whisper_model, align_model, metadata, diarization_pipeline
    whisper_model = whisperx.load_model(settings.WHISPER_MODEL, DEVICE, compute_type="int8")
    align_model, metadata = whisperx.load_align_model(language_code=None, device=DEVICE)
    diarization_pipeline = Pipeline.from_pretrained(settings.PYANNOTE_PROTOCOL, use_auth_token=settings.HUGGINGFACE_TOKEN)

class TaskResponse(BaseModel):
    task_id: str

@app.post("/transcribe", response_model=TaskResponse)
async def transcribe(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Unsupported Media Type")
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"]:
        raise HTTPException(status_code=415, detail="Unsupported file extension")

    date_folder = datetime.utcnow().strftime("%Y-%m-%d")
    folder = os.path.join(settings.UPLOAD_FOLDER, date_folder)
    os.makedirs(folder, exist_ok=True)

    temp_path = os.path.join(folder, f"tmp_{filename}")
    size = 0
    try:
        import aiofiles
        async with aiofiles.open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(1024*1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.MAX_FILE_SIZE:
                    await out.close()
                    os.remove(temp_path)
                    raise HTTPException(status_code=413, detail="File too large")
                await out.write(chunk)
    except OSError as e:
        logging.error(f"Disk error: {e}", exc_info=True)
        cleanup_files.delay()
        raise HTTPException(status_code=503, detail="Server storage full, try later")

    final_name = f"{uuid.uuid4()}_{filename}"
    final_path = os.path.join(folder, final_name)
    os.replace(temp_path, final_path)

    task = transcribe_task.delay(final_path)
    return TaskResponse(task_id=task.id)

@app.get("/result/{task_id}")
def result(task_id: str):
    res = AsyncResult(task_id)
    if res.status == 'SUCCESS':
        data = res.result or {}
        file_path = data.get('file_path')
        if file_path:
            labels_path = os.path.splitext(file_path)[0] + "_labels.json"
            if os.path.exists(labels_path):
                mapping = json.load(open(labels_path))
                for seg in data.get('diarization', []):
                    seg['speaker'] = mapping.get(seg['speaker'], seg['speaker'])
        return JSONResponse({"task_id": task_id, "status": res.status, "result": data})
    elif res.status == 'PENDING':
        return JSONResponse({"task_id": task_id, "status": res.status})
    else:
        return JSONResponse({"task_id": task_id, "status": res.status, "result": str(res.result)})

@app.post("/hooks/tus")
async def tus_hook(request: Request):
    data = await request.json()
    file_id = data.get("FileID")
    if not file_id:
        raise HTTPException(status_code=400, detail="No FileID")
    date_folder = datetime.utcnow().strftime("%Y-%m-%d")
    folder = os.path.join(settings.UPLOAD_FOLDER, date_folder)
    os.makedirs(folder, exist_ok=True)
    src = os.path.join(settings.UPLOAD_FOLDER, file_id)
    dst = os.path.join(folder, f"{uuid.uuid4()}_{secure_filename(file_id)}")
    os.replace(src, dst)
    transcribe_task.delay(dst)
    return {"status":"ok"}

@app.get("/snippet/{task_id}")
def snippet(task_id: str, start: float, end: float):
    path = get_file_path_by_task_id(task_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    out, _ = (
        ffmpeg
        .input(path, ss=start, to=end)
        .output("pipe:", format=settings.SNIPPET_FORMAT)
        .run(capture_stdout=True)
    )
    return Response(content=out, media_type=f"audio/{settings.SNIPPET_FORMAT}")

class LabelRequest(BaseModel):
    mapping: dict

@app.post("/label/{task_id}")
def label(task_id: str, req: LabelRequest):
    path = get_file_path_by_task_id(task_id)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    meta_path = os.path.splitext(path)[0] + "_labels.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(req.mapping, f, ensure_ascii=False, indent=2)
    return {"status":"labels saved"}
