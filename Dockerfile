# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

# Устанавливаем системные зависимости и pip3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Кэшируем установку Python-зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Общие ресурсы приложения
COPY . .

# Точка входа для API
FROM base AS api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Точка входа для CPU-воркера
FROM base AS cpu-worker
CMD ["celery", "-A", "celery_app", "worker", "--concurrency=4", "--queues=preprocess_cpu"]