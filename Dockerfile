# Dockerfile (api + cpu-worker)
FROM python:3.10-slim@sha256:<digest>

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc python3-dev ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# non-root run
RUN useradd --create-home appuser
WORKDIR /app
USER appuser

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl --fail http://127.0.0.1:8000/health || exit 1

# default (API)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]