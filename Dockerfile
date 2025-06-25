# Dockerfile
FROM python:3.10-slim

# 1) system deps, then drop apt cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential \
         ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) copy code
COPY . .

# 4) point all HF‐hub / Transformers / Datasets at our volume
ENV HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache \
    HF_DATASETS_CACHE=/hf_cache

# no CMD here—docker-compose will supply `uvicorn` or `celery` as needed