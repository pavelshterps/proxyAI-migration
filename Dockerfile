# Dockerfile — только API (Uvicorn) и CPU-зависимости
FROM python:3.11-slim

WORKDIR /app

# --- Устанавливаем системные зависимости и чистим кэш ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip build-essential ffmpeg libsndfile1 libpq-dev netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# --- Копируем и устанавливаем «лёгкие» зависимости (CPU) ---
COPY requirements-cpu.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements-cpu.txt && \
    rm -rf /root/.cache/pip

# --- Копируем код приложения ---
COPY . .

# По умолчанию — CPU
ENV WHISPER_DEVICE=cpu

# Запускаем только Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]