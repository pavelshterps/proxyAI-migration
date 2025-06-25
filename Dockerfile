# syntax=docker/dockerfile:1

### Общая база с Python и ffmpeg ###
FROM python:3.10-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

### Стадия для запуска FastAPI ###
FROM base AS api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

### Стадия для CPU-воркера (диаризация) ###
FROM base AS cpu-worker
CMD ["celery", "-A", "celery_app", "worker", "--concurrency", "4", "--queues", "preprocess_cpu"]