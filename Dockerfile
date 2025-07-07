# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 1) Системные зависимости + git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
      ffmpeg \
      build-essential \
      gcc \
      python3-dev \
      git && \
    rm -rf /var/lib/apt/lists/*

# 2) Python-зависимости
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3) Клонируем Hitachi-speech/EEND и настраиваем PYTHONPATH
RUN git clone https://github.com/hitachi-speech/EEND.git /tmp/eend \
 && cp -r /tmp/eend/eend /app/ \
 && rm -rf /tmp/eend
ENV PYTHONPATH="/tmp/eend:${PYTHONPATH}"

# 4) Копируем ваш код
COPY . .

# 5) ФастАПИ-приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]