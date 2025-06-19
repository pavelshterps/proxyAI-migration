# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools required for compiling bitsandbytes and other native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . /app

# Ensure upload directory exists and is writable
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

# Expose the FastAPI port
EXPOSE 8000

# Launch the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]