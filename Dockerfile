FROM python:3.10-slim

# install ffmpeg, python-модулей для статической отдачи
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# копируем только список пакетов
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем код
COPY . .

# запускаем через shell-форму, чтобы подставились $VAR
CMD uvicorn main:app \
    --host $FASTAPI_HOST \
    --port $FASTAPI_PORT \
    --workers $API_WORKERS