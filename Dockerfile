# Dockerfile

# ─── Базовый образ с CUDA и PyTorch ───
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

# Рабочая директория приложения
WORKDIR /app

# Копируем весь проект (включая папку config) в /app
COPY . /app

# Папка для загрузок и права
RUN mkdir -p /tmp/uploads \
 && chmod -R 777 /tmp/uploads

# Чтобы Python видел package config и другие модули
ENV PYTHONPATH=/app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Порт FastAPI
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]