# Dockerfile

# 1) Базовый образ Python
FROM python:3.10-slim

# 2) Переключаем mirror'ы на российские и ставим системные зависимости
RUN sed -i 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' /etc/apt/sources.list \
 && sed -i 's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3-pip \
      ffmpeg \
      build-essential \
      gcc \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

# 3) Рабочая директория
WORKDIR /app

# 4) Устанавливаем Python-зависимости
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5) Копируем код
COPY . .

# 6) По умолчанию — запускаем API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]