# base image
FROM python:3.10-slim

# system deps for webrtcvad / build-essential
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        python3-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]