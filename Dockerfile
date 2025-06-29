# syntax=docker/dockerfile:1
FROM python:3.10-slim

#---- install system deps ----
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      git-lfs \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

#---- application code ----
COPY . .

# default command will be provided by docker-compose per‐service