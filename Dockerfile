FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Установка зависимостей системы (git, ffmpeg, и всё что требуется для работы аудио и Python)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir "ctranslate2[cuda11]"
RUN pip install --no-cache-dir -r requirements.txt

# Копирование проекта
COPY . .

# Разрешение на запись в /tmp/uploads для всех
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

# Убедись что весь код копируется под /app, WORKDIR /app уже установлен

# По умолчанию ничего не запускаем (каждый сервис прописывает свою команду в docker-compose)
CMD [ "bash" ]