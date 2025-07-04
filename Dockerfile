# Dockerfile

# 1) Base image
FROM python:3.10-slim

# 2) Swap in faster Russian mirrors
RUN sed -i \
      -e 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' \
      -e 's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' \
      /etc/apt/sources.list

# 3) System dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3-pip \
        ffmpeg \
        build-essential \
        gcc \
        python3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4) Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5) Copy the rest of your code
COPY . .

# 6) Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]