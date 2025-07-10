FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip build-essential ffmpeg libsndfile1 git \
      libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .
ENV WHISPER_DEVICE=cpu

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--concurrency=1", "--queues=preprocess_cpu"]