FROM python:3.10-slim

# Подставляем российские зеркала вместо deb.debian.org и security.debian.org
RUN sed -i \
      's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' \
      /etc/apt/sources.list.d/debian.sources \
 && sed -i \
      's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' \
      /etc/apt/sources.list.d/debian.sources

# Обновляем пакеты и устанавливаем необходимые зависимости
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip ffmpeg build-essential gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# По умолчанию запускаем API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]