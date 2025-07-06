# Dockerfile

# 1) Base image
FROM python:3.10-slim

# 2) Set workdir
WORKDIR /app

# 3) (Optional) Switch to faster Russian mirrors
RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' /etc/apt/sources.list && \
      sed -i 's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' /etc/apt/sources.list; \
    fi

# 4) System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip ffmpeg build-essential gcc python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# 5) Install Python reqs
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6) Clone & install FS-EEND as a local package
RUN git clone https://github.com/hitachi-speech/EEND.git /app/eend && \
    # inject a minimal setup.py so pip can install it
    cat > /app/eend/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="eend",
    version="0.1.0",
    packages=find_packages(),
)
EOF
    pip install --no-cache-dir /app/eend

# 7) Copy your proxyAI app
COPY . .

# 8) Default command — FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]