# Dockerfile (API + CPU worker)

FROM python:3.11-slim

WORKDIR /app

# --- Install system deps and clean up ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip build-essential ffmpeg libsndfile1 libpq-dev netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# --- Copy and install only the "CPU" Python deps ---
COPY requirements-cpu.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements-cpu.txt && \
    rm -rf /root/.cache/pip

# --- Copy the rest of your code ---
COPY . .

# Default to CPU
ENV WHISPER_DEVICE=cpu

# Run Uvicorn + Celery CPU worker
CMD ["sh", "-c", "\
    uvicorn main:app --host 0.0.0.0 --port 8000 & \
    celery -A tasks worker --loglevel=info --concurrency=1 --queues=split_cpu,dispatch_cpu,collect_cpu \
"]