import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from celery.result import AsyncResult
import config.settings as settings
from tasks import diarize_full, transcribe_segments

app = FastAPI()

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/transcribe")
async def submit_transcription(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["wav", "mp3", "m4a", "flac"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    uid = str(uuid.uuid4())
    dest = os.path.join(settings.UPLOAD_FOLDER, f"{uid}.{ext}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(await file.read())

    # Chain tasks: diarize_full -> transcribe_segments
    chain = diarize_full.s(dest) | transcribe_segments.s(dest)
    result = chain.apply_async()
    return {"task_id": result.id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    res = AsyncResult(task_id)
    if not res.ready():
        return JSONResponse({"status": "PENDING"}, status_code=202)
    if res.failed():
        return JSONResponse(
            {"status": "FAILURE", "error": str(res.result)},
            status_code=500
        )
    return {"status": "SUCCESS", "data": res.result}