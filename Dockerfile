# Dockerfile
FROM python:3.10-slim

# Install system dependencies for audio processing + C‐extensions build (webrtcvad, Julius, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       build-essential \            # gcc, make, etc. for building wheels  [oai_citation:1‡forum-nas.fr](https://www.forum-nas.fr/threads/bazarr-1-1-4-0-companion-application-to-sonarr-and-radarr-it-manages-and-downloads-subtitles.10638/page-4?utm_source=chatgpt.com)
       python3-dev \                # Python C headers
       libsndfile1 \                # needed by pydub / soundfile
       libopus-dev \                # optional, for advanced audio codecs
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]