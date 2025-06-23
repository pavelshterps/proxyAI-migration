# Dockerfile
FROM python:3.10-slim

# Установим системные зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Переменная для кеша HF
ARG HUGGINGFACE_TOKEN

# Префетчим Pyannote-диаризацию в кэш
RUN pip3 install --no-cache-dir pyannote.audio huggingface-hub
RUN python3 - <<EOF
from pyannote.audio import Pipeline
import os
Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
    cache_dir="/hf_cache"
)
EOF

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]