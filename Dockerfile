FROM python:3.10-slim

# Install OS deps, then clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3-pip ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

ENV PYTHONUNBUFFERED=1

# Default to running the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]