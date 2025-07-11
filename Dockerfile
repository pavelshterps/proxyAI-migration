FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing and PostgreSQL driver
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      build-essential \
      ffmpeg \
      libsndfile1 \
      git \
      libavformat-dev \
      libavcodec-dev \
      libavutil-dev \
      libswscale-dev \
      libportaudio2 \
      libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies, including psycopg2-binary for PostgreSQL
COPY requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt psycopg2-binary

# Copy application code
COPY . .

# Default to CPU Whisper device
ENV WHISPER_DEVICE=cpu

# Start Celery worker on the CPU queue
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--concurrency=1", "--queues=preprocess_cpu"]