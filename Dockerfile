# syntax=docker/dockerfile:1

########################################################
# Dockerfile for API server and CPU worker
########################################################

# Use official slim Python 3.10 base image
FROM python:3.10-slim

# Install system dependencies:
#  - build-essential, gcc, python3-dev → for compiling any C-extensions (webrtcvad, julius, etc.)
#  - ffmpeg → for audio processing
#  - ca-certificates → for HTTPS model downloads (Hugging Face, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      python3-dev \
      ffmpeg \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create and switch to a non-root user (optional but recommended)
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
USER appuser

# Copy requirements and install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Default command: start FastAPI as the API server.
# In docker-compose.yml, for cpu-worker and gpu-worker you will override this:
#   cpu-worker → celery -A celery_app worker --loglevel=info --queues preprocess_cpu
#   gpu-worker → celery -A celery_app worker --loglevel=info --queues preprocess_gpu
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]