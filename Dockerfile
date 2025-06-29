# Dockerfile
FROM python:3.10.13-slim-bullseye@sha256:…  # pin digest
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg git-lfs \
  && git lfs install \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home proxy && chown -R proxy:proxy /app
USER proxy
ENV PATH="/home/proxy/.local/bin:${PATH}"