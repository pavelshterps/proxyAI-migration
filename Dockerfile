# Dockerfile
FROM python:3.10-slim AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]