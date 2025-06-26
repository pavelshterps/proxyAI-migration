# Dockerfile
FROM python:3.10-slim as base

# system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# default entrypoint (API)
CMD ["uvicorn", "main:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--workers", "1"]