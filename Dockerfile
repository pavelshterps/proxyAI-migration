# syntax=docker/dockerfile:1
FROM python:3.10-slim

# — системные зависимости
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.10 python3-pip ffmpeg build-essential gcc python3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# — кешируем установку Python-зависимостей
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# — копируем весь код
COPY . .

# — по-умолчанию запускаем Celery Beat
ENTRYPOINT ["celery"]
CMD ["-A", "celery_app", "beat", "--loglevel=info"]