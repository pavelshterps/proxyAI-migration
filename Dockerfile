# Dockerfile (multi‐stage)

### Stage 1: build all Python deps ###
FROM python:3.10-slim AS builder
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

### Stage 2a: API & CPU worker ###
FROM python:3.10-slim AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
ENV UPLOAD_FOLDER=/tmp/uploads HF_HOME=/hf_cache
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

### Stage 2b: GPU worker ###
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
ENV UPLOAD_FOLDER=/tmp/uploads HF_HOME=/hf_cache
ENTRYPOINT ["celery", "-A", "celery_app.celery_app", "worker"]