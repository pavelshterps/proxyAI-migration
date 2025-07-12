# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# --- Устанавливаем системные зависимости и чистим apt-кэш ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip build-essential ffmpeg libsndfile1 libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Копируем и ставим только необходимые для API и CPU-воркера Python-пакеты ---
COPY requirements.txt ./

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
      fastapi uvicorn[standard] celery redis python-dotenv \
      python-multipart pydantic pydantic-settings \
      structlog sqlalchemy[asyncio] asyncpg aiosqlite \
      prometheus-client slowapi limits yamlargparse chainer numpy<2.0 \
      webrtcvad pydub librosa soundfile httpx sse-starlette \
      psycopg2-binary && \
    rm -rf /root/.cache/pip

# --- Копируем код приложения ---
COPY . .

# По умолчанию работаем на CPU
ENV WHISPER_DEVICE=cpu

# Запускаем одновременно Uvicorn и Celery-воркер для CPU-очередей
CMD ["sh", "-c", "\
    uvicorn main:app --host 0.0.0.0 --port 8000 & \
    celery -A tasks worker --loglevel=info --concurrency=1 --queues=split_cpu,dispatch_cpu,collect_cpu \
"]