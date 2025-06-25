FROM python:3.10-slim

COPY static ./static
WORKDIR /app

# системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

#python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# приложение
COPY . .

# команда по умолчанию (docker-compose её переопределит)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]