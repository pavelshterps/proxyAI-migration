from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
import uuid
import os

from celery_app import celery_app
from settings import FASTAPI_HOST, FASTAPI_PORT, UPLOAD_FOLDER, TUS_ENDPOINT
from tasks import transcribe_task

app = FastAPI()

# раздаём загруженные файлы и tusd-файлы по единому пути
app.mount("/files", StaticFiles(directory=UPLOAD_FOLDER), name="files")

@app.post("/transcribe")
async def api_transcribe(file: UploadFile):
    # сохраняем файл
    ext = os.path.splitext(file.filename)[1]
    dest = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}{ext}")
    with open(dest, "wb") as f:
        f.write(await file.read())

    # запускаем фоновую задачу
    result = transcribe_task.delay(dest)
    return JSONResponse({
        "task_id": result.id,
        "estimate": None  # тут фронт сразу подставит своё UI
    }, status_code=202)

@app.get("/result/{task_id}")
def api_result(task_id: str):
    """Возвращает статус, а когда merge_chunks отработает — JSON с segments и audio_filepath."""
    merge_id = AsyncResult(task_id, app=celery_app).get(propagate=False)
    if isinstance(merge_id, dict):
        # это сразу результат merge_chunks
        return merge_id
    else:
        res = AsyncResult(task_id, app=celery_app)
        return {"status": res.status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=FASTAPI_HOST, port=FASTAPI_PORT, workers=1)