import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from celery_app import celery
from config.settings import settings
from tasks import estimate_processing_time

app = FastAPI()

# Serve the main index page at root
@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve the main index page
    index_path = os.path.join("static", "index.html")
    return HTMLResponse(content=open(index_path, encoding="utf-8").read())

# Serve static files under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def start_transcription(file: UploadFile = File(...)):
    """
    Принимает аудиофайл, сохраняет и запускает Celery-таску.
    Возвращает merge_task_id для последующего polling и estimate.
    """
    # Сохраняем файл
    upload_folder = settings.UPLOAD_FOLDER
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {e}")

    # Запускаем task, который создаёт chord и возвращает merge_task_id
    task = celery.send_task("tasks.transcribe_task", args=[file_path])
    estimate = estimate_processing_time(file_path)
    return JSONResponse({"merge_task_id": task.id, "estimate": estimate})

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Проверка состояния merge-задачи.
    Если SUCCESS, возвращаем уже готовый результат с сегментами.
    """
    res = celery.AsyncResult(task_id)
    state = res.state

    if state == "SUCCESS":
        payload = res.get()
        # payload = {"segments": [...], "audio_filepath": "..."}
        return {"status": state, **payload}

    if state in ("PENDING", "STARTED", "RETRY"):
        return {"status": state}

    # FAILURE и прочие
    info = res.info
    return {"status": state, "error": str(info)}

# Приложение раздаёт index.html автоматически через StaticFiles