FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8000
USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "${API_WORKERS}"]
