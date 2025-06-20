# Базовый образ с поддержкой CUDA и PyTorch
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Устанавливаем системные зависимости для bitsandbytes, whisperx, pydub и ffmpeg
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      ffmpeg \
      libsndfile1 \
      tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала копируем только requirements, чтобы закешировать зависимости
COPY requirements.txt /app/
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Прогреваем кеш моделей выравнивания для английского и русского
RUN python - << 'EOF'
import whisperx
for lang in ("english","russian"):
    # Preload alignment models by positional args to match API signature
    whisperx.load_align_model(
        "whisper-large",
        "cpu",
        lang,
        5
    )
EOF

# Копируем весь код
COPY . /app

# Делаем папку для загрузок
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

# Открываем порт FastAPI
EXPOSE 8000

# Запуск по-умолчанию (API)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]