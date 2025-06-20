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

# Pre-warm whisperx alignment models for English and Russian
RUN python - << 'EOF'
import os, whisperx
# read settings from env (fall back to these defaults)
ALIGN_MODEL = os.getenv("ALIGN_MODEL_NAME", "jonatasgrosman/wav2vec2-large-xlsr-53-english")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
BEAM = int(os.getenv("ALIGN_BEAM_SIZE", "5"))
for lang in ("english", "russian"):
    try:
        whisperx.load_align_model(
            ALIGN_MODEL,
            DEVICE,
            COMPUTE,
            lang,
            BEAM
        )
    except Exception:
        pass
EOF

# Копируем весь код
COPY . /app

# Делаем папку для загрузок
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

# Открываем порт FastAPI
EXPOSE 8000

# Запуск по-умолчанию (API)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]