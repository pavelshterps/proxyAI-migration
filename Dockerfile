FROM python:3.10-slim

# Установим инструменты сборки и runtime-зависимости для pydub/ffmpeg и bitsandbytes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установим Python-зависимости
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем приложение
COPY . /app

# Папка для загрузок
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]