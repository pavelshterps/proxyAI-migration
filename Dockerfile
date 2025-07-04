# Dockerfile

FROM python:3.10-slim

# системные зависимости
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip ffmpeg build-essential gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# сначала зависимые библиотеки
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# теперь весь код
COPY . .

# по умолчанию — запускаем API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]