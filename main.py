import os
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
import aiofiles

from celery_app import celery_app
from config.settings import UPLOAD_FOLDER, FASTAPI_PORT, MAX_FILE_SIZE
from tasks import diarize_full

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join("static", "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=500, detail="Index not found")
    return HTMLResponse(open(index_path, "r", encoding="utf-8").read())

@app.post("/transcribe", status_code=202)
async def start_transcription(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_FOLDER, filename)
    async with aiofiles.open(dest_path, 'wb') as out_file:
        await out_file.write(data)
    task = diarize_full.apply_async((dest_path,), queue="preprocess_cpu")
    logger.info("Submitted diarization task %s for file %s", task.id, dest_path)
    return JSONResponse({"task_id": task.id})

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    ar = AsyncResult(task_id, app=celery_app)
    state = ar.state
    if state in ("PENDING", "STARTED"):
        return JSONResponse({"status": state}, status_code=202)
    if state == "FAILURE":
        return JSONResponse({"status": "FAILURE", "error": str(ar.result)}, status_code=500)

    result = ar.result
    if isinstance(result, dict):
        return {"status": "SUCCESS", **result}
    if isinstance(result, list):
        return {"status": "SUCCESS", "segments": result}
    return {"status": "SUCCESS", "data": result}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=FASTAPI_PORT)