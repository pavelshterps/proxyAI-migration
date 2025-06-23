import os, uuid, logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
from celery_app import celery_app
from config.settings import UPLOAD_FOLDER, FASTAPI_PORT, MAX_FILE_SIZE_MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join("static", "index.html")
    if not os.path.isfile(path):
        raise HTTPException(500, f"Index not found")
    return HTMLResponse(open(path).read())

@app.post("/transcribe", status_code=202)
async def start(file: UploadFile = File(...), background: BackgroundTasks = None):
    data = await file.read()
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    dest = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    with open(dest, "wb") as f:
        f.write(data)
    if background:
        background.add_task(lambda p=dest: os.remove(p))
    job = celery_app.send_task("tasks.transcribe_full", args=[dest])
    logger.info("Submitted task %s", job.id)
    return JSONResponse({"task_id": job.id})

@app.get("/result/{task_id}")
async def result(task_id: str):
    ar = AsyncResult(task_id, app=celery_app)
    st = ar.state
    if st in ("PENDING","STARTED"):
        return JSONResponse({"status": st}, 202)
    if st == "SUCCESS":
        data = ar.get()
        # merge_results now returns {"text":...}
        text = data.get("text") if isinstance(data, dict) else str(data)
        return {"status": st, "text": text}
    return JSONResponse({"status": st, "error": str(ar.info)}, 500)