FROM python:3.10-slim AS base

# убираем лишние рекомендации и чистим кэш
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# чтобы HuggingFace hub кешировал не в /root/.cache, а в hf_cache-том
ENV HF_HOME=/hf_cache