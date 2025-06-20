# config/settings.py

import os
import torch

# Папка для загруженных файлов
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")

# Настройки FastAPI
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

# Проверяем доступность GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Если есть CUDA — загружаем модель в 8-битном режиме через bitsandbytes
LOAD_IN_8BIT = torch.cuda.is_available()

# Параметры моделей и токен
WHISPER_MODEL_NAME   = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

ALIGN_MODEL_NAME     = os.getenv(
    "ALIGN_MODEL_NAME",
    "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
)
ALIGN_BEAM_SIZE      = int(os.getenv("ALIGN_BEAM_SIZE", "5"))

HUGGINGFACE_TOKEN    = os.getenv("HUGGINGFACE_TOKEN", None)