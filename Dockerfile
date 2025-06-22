# syntax=docker/dockerfile:1.3

############################
# Stage 1: build dependencies
############################
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libsndfile1 \
      ffmpeg \
      git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies into the Conda environment
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

############################
# Stage 2: runtime
############################
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Create non-root user with home directory and set permissions
RUN groupadd -r appuser \
 && useradd -r -g appuser -m -d /home/appuser appuser \
 && mkdir -p /home/appuser /tmp/uploads /tmp/chunks /tmp/hf_cache \
 && chown -R appuser:appuser /home/appuser /tmp/uploads /tmp/chunks /tmp/hf_cache

# Force caches into writable tmp dirs
ENV HOME=/home/appuser
ENV MPLCONFIGDIR=/tmp
ENV HF_HOME=/tmp/hf_cache

WORKDIR /app

# Copy installed packages and console scripts from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

# Copy application code and static assets
COPY . /app

# Ensure index.html is in place for root endpoint
RUN cp static/index.html index.html \
 && chown -R appuser:appuser /app

USER appuser
ENV PATH="/opt/conda/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app",
     "--host", "0.0.0.0", "--port", "8000",
     "--graceful-timeout", "30", "--limit-max-requests", "1000"]