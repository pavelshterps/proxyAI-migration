# Используем официальный образ PyTorch с поддержкой CUDA (или замените на CPU-образ, если нужно)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Добавляем непривилегированного пользователя
RUN useradd --create-home appuser

# Уже есть создание пользователя appuser
RUN mkdir -p /tmp/uploads && chmod -R 777 /tmp/uploads

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       ffmpeg \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей и ставим Python-библиотеки
COPY requirements.txt .
RUN pip install --no-cache-dir "ctranslate2[cuda11]" \
    && pip install --no-cache-dir -r requirements.txt



# Копируем весь код приложения
COPY . .

# Переключаемся на непривилегированного пользователя
USER appuser