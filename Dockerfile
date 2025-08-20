FROM python:3.11-slim

WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}
ENV PIP_NO_PROGRESS_BAR=1

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt ./requirements-cpu.txt
RUN pip install -q --upgrade pip && pip install -q --no-cache-dir -r requirements.txt

COPY . .

# Гарантируем, что этот воркер точно не пойдёт в CUDA
ENV WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8

CMD ["celery", "-A", "tasks", "worker", \
     "--loglevel=info", \
     "--concurrency=2", \
     "--queues=webhooks"]