# Dockerfile
FROM python:3.10-slim

ENV UPLOAD_FOLDER=/tmp/uploads

# cache & build-time model dir
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/1

WORKDIR /app

# system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc \
      ffmpeg \
      git \
 && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-cache pyannote diarization model
RUN python - << 'EOF'
from pyannote.audio import Pipeline
import os
Pipeline.from_pretrained(
    os.getenv("PYANNOTE_PROTOCOL"),
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
    cache_dir=os.getenv("HF_HOME")
)
EOF

# copy project
COPY . .

# ensure upload folder exists
RUN mkdir -p ${UPLOAD_FOLDER}

EXPOSE 8000
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]