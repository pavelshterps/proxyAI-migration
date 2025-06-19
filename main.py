import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult

from celery_app import celery
from config.settings import settings
from tasks import estimate_processing_time

app = FastAPI()

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join("static", "index.html")
    return HTMLResponse(content=open(index_path, encoding="utf-8").read())

# Static files (JS/CSS/images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# serve uploaded files by URL /files/{filename}
app.mount(
    "/files",
    StaticFiles(directory=settings.UPLOAD_FOLDER),
    name="uploads"
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def start_transcription(file: UploadFile = File(...)):
    # Save uploaded file with unique name
    upload_folder = settings.UPLOAD_FOLDER
    os.makedirs(upload_folder, exist_ok=True)
    unique_id = uuid.uuid4().hex
    safe_name = file.filename.replace(" ", "_")
    unique_filename = f"{unique_id}_{safe_name}"
    file_path = os.path.join(upload_folder, unique_filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {e}")

    # Dispatch the wrapper task (creates and dispatches chord)
    wrapper = celery.send_task("tasks.transcribe_task", args=[file_path])
    estimate = estimate_processing_time(file_path)
    return JSONResponse({"task_id": wrapper.id, "estimate": estimate})

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    # First check wrapper task
    wrapper = AsyncResult(task_id)
    w_state = wrapper.state

    if w_state == "SUCCESS":
        # wrapper payload contains merge_task_id
        w_payload = wrapper.get()
        merge_id = w_payload.get("merge_task_id")
        # if merge step exists, poll it
        if merge_id:
            merge = AsyncResult(merge_id)
            m_state = merge.state
            if m_state == "SUCCESS":
                final = merge.get()
                # convert disk path to URL
                filename = os.path.basename(final.get("audio_filepath", ""))
                final["audio_filepath"] = f"/files/{filename}"
                return {"status": m_state, **final}
            if m_state in ("PENDING", "STARTED", "RETRY"):  # still working
                return {"status": m_state}
            # merge failed
            return {"status": m_state, "error": str(merge.info)}
        # no merge, return wrapper payload directly
        return {"status": w_state, **w_payload}

    if w_state in ("PENDING", "STARTED", "RETRY"):
        return {"status": w_state}

    # wrapper failed
    return {"status": w_state, "error": str(wrapper.info)}