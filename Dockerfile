FROM python:3.11-slim

WORKDIR /app

# 1) Неинтерактивный режим для apt
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

# 2) Системные зависимости в тихом режиме
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      python3-pip build-essential ffmpeg libsndfile1 libpq-dev netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# 3) Настроим pip на тихий режим
ENV PIP_NO_PROGRESS_BAR=1

# 4) “Лёгкие” зависимости (CPU)
COPY requirements-cpu.txt ./
RUN pip3 install --upgrade pip -q && \
    pip3 install -q --no-cache-dir --progress-bar off -r requirements-cpu.txt && \
    rm -rf /root/.cache/pip

# 5) Копируем код приложения
COPY . .

# По умолчанию — CPU
ENV WHISPER_DEVICE=cpu

# 6) Запуск Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]