# proxyAI v13.6 – Dockerfile

FROM python:3.10-slim

# Install system deps (build-essential for webrtcvad, ffmpeg for audio)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc python3-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Upgrade pip, install pure-Python deps
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default entrypoint is uvicorn for the API,
# but overridden in docker-compose for celery workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]