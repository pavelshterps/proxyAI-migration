# Dockerfile — для api и CPU-воркера

FROM python:3.11-slim

WORKDIR /app

# 1) Системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip build-essential ffmpeg libsndfile1 libpq-dev netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# 2) Лёгкие питон-зависимости для API и CPU
COPY requirements-cpu.txt ./
ENV PIP_NO_PROGRESS_BAR=off
RUN pip3 install --upgrade pip && \
    pip3 install -q --no-cache-dir --progress-bar off -r requirements-cpu.txt && \
    rm -rf /root/.cache/pip

# 3) Копируем приложение
COPY . .

# По умолчанию — CPU
ENV WHISPER_DEVICE=cpu

# 4) Запуск API + CPU-воркера
CMD ["sh", "-c", "\
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${API_WORKERS} & \
    celery -A tasks worker \
      --loglevel=info \
      --concurrency=${CPU_CONCURRENCY} \
      --queues=split_cpu,dispatch_cpu,collect_cpu \
"]