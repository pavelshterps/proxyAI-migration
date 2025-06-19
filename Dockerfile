# Use the official PyTorch CUDA image so bitsandbytes can build its GPU kernels
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Suppress tzdata prompts and set timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system libraries for audio processing, pydub/ffmpeg and bitsandbytes build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 # Install bitsandbytes (GPU-enabled) after torch is present
 && pip install --no-cache-dir bitsandbytes

# Copy the rest of the application code
COPY . /app

# Ensure upload directory exists and is writable
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

# Expose FastAPI port
EXPOSE 8000

# Launch the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]