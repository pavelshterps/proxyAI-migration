import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from celery.result import AsyncResult

from celery_app import celery_app
from settings import UPLOAD_FOLDER, FASTAPI_HOST, FASTAPI_PORT

app = FastAPI()

# Статика для файловых отдач
app.mount("/files", StaticFiles(directory=UPLOAD_FOLDER), name="files")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def start_transcription(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    dest = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    with open(dest, "wb") as f:
        f.write(await file.read())

    job = celery_app.send_task("tasks.transcribe_task", args=[dest])
    # estimate можно возвращать из задачи, но для UI — сразу ноль
    return JSONResponse({"task_id": job.id, "estimate": 0}, status_code=202)

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    wrapper = AsyncResult(task_id, app=celery_app)
    state = wrapper.state

    if state == "SUCCESS":
        payload = wrapper.get()
        merge_id = payload.get("merge_task_id")
        if merge_id:
            merge = AsyncResult(merge_id, app=celery_app)
            if merge.state == "SUCCESS":
                final = merge.get()
                return final
            return {"status": merge.state}
        return {"status": state}

    if state in ("PENDING", "STARTED", "RETRY"):
        return {"status": state}

    return {"status": state, "error": str(wrapper.info)}