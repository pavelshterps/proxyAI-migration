# Dockerfile

FROM python:3.10-slim
WORKDIR /app

# Российские зеркала (по желанию)
RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' /etc/apt/sources.list && \
      sed -i 's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' /etc/apt/sources.list; \
    fi

# Системные пакеты
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip ffmpeg build-essential gcc python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#  — FS-EEND dependencies —
RUN pip install --no-cache-dir yamlargparse chainer numpy<2.0

# Vendored FS-EEND
COPY eend/ /app/eend/
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Копируем всё приложение
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]