# === stage 1: build deps ===
FROM python:3.10-slim AS builder
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ffmpeg git && \
    pip install --no-cache-dir poetry==1.8.1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without-hashes > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# === stage 2: final ===
FROM python:3.10-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "${API_WORKERS:-1}"]