# Используем официальный образ PyTorch с поддержкой CUDA (или замените на CPU-образ, если нужно)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Добавляем непривилегированного пользователя
RUN useradd --create-home appuser

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

# Правим импорты в whisperx и transformers для совместимости
RUN if [ -f "$(python -c "import whisperx, os; print(os.path.join(os.path.dirname(whisperx.__file__), 'asr', 'transforms.py'))")" ]; then \
      sed -i "s|from transformers import Pipeline|from transformers.pipelines.base import Pipeline|" "$(python -c "import whisperx, os; print(os.path.join(os.path.dirname(whisperx.__file__), 'asr', 'transforms.py'))")"; \
    else \
      echo "whisperx transforms.py not found, skipping patch"; \
    fi

RUN sed -i \
      -e "s/NEAREST_EXACT/NEAREST/" \
      -e "s/BICUBIC_EXACT/BICUBIC/" \
      $(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'models', 'common_vision.py'))")

# Копируем весь код приложения
COPY . .

# Переключаемся на непривилегированного пользователя
USER appuser