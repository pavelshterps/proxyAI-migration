# базовый образ со slim-Python
FROM python:3.10-slim

# собрать системные зависимости
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc python3-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# копируем pip-файл и ставим зависимости
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# копируем всё приложение
COPY . .

# дефолтная точка входа (api)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]