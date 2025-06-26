FROM python:3.10-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
# Copy configuration, code, and static frontend
COPY config        ./config
COPY main.py       ./main.py
COPY celery_app.py ./celery_app.py
COPY tasks.py      ./tasks.py
COPY static        ./static




COPY . .
ENV PYTHONUNBUFFERED=1