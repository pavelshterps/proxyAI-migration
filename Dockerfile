# syntax=docker/dockerfile:1
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc python3-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

COPY . .

# Production ASGI via Gunicorn+UvicornWorker
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker -w ${API_WORKERS} -b 0.0.0.0:8000"]