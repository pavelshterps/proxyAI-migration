FROM continuumio/miniconda3

# Install OS-level dependencies for audio processing and image support
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Создание conda-окружения
COPY environment.yml .
RUN conda env create -f environment.yml

# Активируем окружение в последующих командах
SHELL ["conda", "run", "-n", "proxyai", "/bin/bash", "-c"]

# Копируем исходники и устанавливаем Python-зависимости
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Добавляем conda-окружение в PATH
ENV PATH /opt/conda/envs/proxyai/bin:$PATH

# Открываем порт и запускаем FastAPI
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]