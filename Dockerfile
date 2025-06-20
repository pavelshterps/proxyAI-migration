# ─── Базовый образ с поддержкой CUDA и PyTorch ───
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Отключаем интерактивные запросы и настраиваем часовой пояс
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      build-essential \
      cmake \
      git \
      ffmpeg \
      libsndfile1 \
 && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Устанавливаем Python-зависимости ───
COPY requirements.txt /app/
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ─── Копируем весь код проекта ───
COPY . /app

# ─── Прогреваем кеш whisperx-align моделей для English и Russian ───
RUN python - << 'EOF'
import whisperx
for lang in ("english","russian"):
    try:
        whisperx.load_align_model(
            "whisper-large",  # ALIGN_MODEL_NAME по умолчанию
            "cpu",            # кешируем на CPU
            lang,
            5                 # ALIGN_BEAM_SIZE по умолчанию
        )
    except Exception:
        pass
EOF

# ─── Папка для загрузок ───
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

# ─── Чтобы Python видел модули в /app ───
ENV PYTHONPATH=/app

# ─── Экспонируем порт FastAPI ───
EXPOSE 8000

# ─── Запуск FastAPI ───
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]