import os
import shutil
from uuid import uuid4

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import RedirectResponse

from celery_app import celery_app
from config.settings import settings

app = FastAPI()

@app.get("/", include_in_schema=False)
async def root():
    """
    Redirect root URL to interactive API docs.
    """
    return RedirectResponse(url="/docs")

@app.post("/transcribe")
async def start_transcription(file: UploadFile):
    """
    Receive an uploaded audio file, save it, and enqueue a Celery task.
    Returns a task_id which can be polled for results.
    """
    # Generate a unique task ID
    task_id = str(uuid4())

    # Build destination path
    upload_folder = settings.UPLOAD_FOLDER
    dest_path = os.path.join(upload_folder, f"{task_id}.wav")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Save the uploaded file
    try:
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Enqueue the diarization+transcription task under our task_id
    celery_app.send_task(
        name="tasks.diarize_full",
        args=(dest_path,),
        task_id=task_id
    )

    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Poll for the result of a given task_id.
    - If still pending, returns status 202 with PENDING.
    - If done, returns status 200 with SUCCESS and the list of segments.
    """
    result = celery_app.backend.get(task_id)
    if result is None:
        # Task still in progress
        return {"status": "PENDING"}, 202

    # Task completed successfully
    return {"status": "SUCCESS", "segments": result}