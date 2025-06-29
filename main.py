import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from uuid import uuid4

from config.settings import settings
from celery_app import celery_app
from routes import router

app = FastAPI()

# mount your application router
app.include_router(router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}