# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Устанавливаем только нужные пакеты, очищаем кеш apt
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ffmpeg \
         git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Копируем и ставим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем весь код и настраиваем рабочую директорию
WORKDIR /app
COPY . .

# По умолчанию команда подменяется в docker-compose.yml