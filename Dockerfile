# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV HF_HOME=/hf_cache
ENV UPLOAD_FOLDER=/uploads