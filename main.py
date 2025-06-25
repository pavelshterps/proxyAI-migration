import os
import shutil
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from celery_app import celery_app
from config.settings import settings

app = FastAPI()

# статика (index.html + JS/CSS) должна лежать в ./static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    path = os.path.join("static", "index.html")
    if os.path.isfile(path):
        return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Not Found")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    if file.content_type != "audio/wav":
        raise HTTPException(400, "Only WAV files supported")
    uid = str(__import__("uuid").uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    celery_app.send_task("tasks.diarize_full", args=(dest,))
    return {"job_id": uid}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = celery_app.backend.get(job_id)
    if result is None:
        return {"status": "PENDING"}
    return {"status": "SUCCESS", "segments": result}