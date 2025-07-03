FROM python:3.10-slim

# 1) Подменяем зеркала на российские (Яндекс)
RUN sed -i 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' /etc/apt/sources.list \
    && sed -i 's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' /etc/apt/sources.list

# 2) Устанавливаем зависимости
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        ffmpeg \
        build-essential \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

# 3) Устанавливаем Python-зависимости
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4) Копируем приложение
COPY . .

# 5) По умолчанию запускаем API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]