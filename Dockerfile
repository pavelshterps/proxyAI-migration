FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# —————— ENV-переменные ——————
# сюда складываем все необходимые переменные окружения,
# например API_WORKERS и ALLOWED_ORIGINS:
ENV ALLOWED_ORIGINS='["*"]'
ENV TORCHVISION_DISABLE_IMAGE_EXTENSIONS=1
ENV API_WORKERS=1




# 1) создаём непривилегированного пользователя
RUN useradd --create-home appuser

# 2) системные зависимости
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# 3) ставим Python-зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir "ctranslate2[cuda11]"   \
 && pip install --no-cache-dir -r requirements.txt

RUN sed -i \
  "s|from transformers import Pipeline|from transformers.pipelines.base import Pipeline|" \
  $(python -c "import whisperx, os; print(os.path.join(os.path.dirname(whisperx.__file__), 'asr.py'))")

RUN sed -i \
      -e "s/NEAREST_EXACT/NEAREST/" \
      -e "s/BICUBIC_EXACT/BICUBIC/" \
      $(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__),'image_utils.py'))")
# 4) копируем код
COPY . .

# 5) переключаемся на небезопасного пользователя
USER appuser

# 6) выставляем порт
EXPOSE 8000

# 7) команда запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "${API_WORKERS}"]
