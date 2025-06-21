# syntax=docker/dockerfile:1.3
# ─── Stage 1: сборка зависимостей ───
FROM python:3.10-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libsndfile1 \
      ffmpeg \
      git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Используем BuildKit cache для pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: runtime с поддержкой CUDA ───
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime-slim

# создаём непривилегированного пользователя
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# копируем зависимости и код
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app

# права на директорию загрузок
RUN mkdir -p /tmp/uploads /tmp/chunks \
 && chown -R appuser:appuser /tmp/uploads /tmp/chunks /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]