# Dockerfile.gpu
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# 1) System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-dev build-essential ffmpeg libsndfile1 git \
      libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

# 2) Common Python deps
COPY requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# 3) GPU-specific packages
RUN pip3 install --no-cache-dir \
      faster-whisper[cuda12]==1.1.1 \
      ctranslate2[cuda12]>=4.6.0

# 4) Application code
COPY . .

# 5) GPU worker env
ENV WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    HUGGINGFACE_CACHE_DIR=/hf_cache \
    CELERY_BROKER_URL=${CELERY_BROKER_URL} \
    UPLOAD_FOLDER=/app/uploads \
    RESULTS_FOLDER=/app/results

# 6) Use shell entrypoint so 'celery' is on PATH under NVIDIA image
ENTRYPOINT ["sh", "-c"]
CMD ["exec celery -A tasks worker --loglevel=info --concurrency=1 --queues=transcribe_gpu,diarize_gpu"]