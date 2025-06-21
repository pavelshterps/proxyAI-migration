import os
import torch

# Celery
BROKER_URL        = os.getenv('CELERY_BROKER_URL',    'redis://redis:6379/0')
RESULT_BACKEND    = os.getenv('CELERY_RESULT_BACKEND','redis://redis:6379/1')

# FastAPI
UPLOAD_FOLDER     = os.getenv('UPLOAD_FOLDER',        '/tmp/uploads')
FASTAPI_HOST      = os.getenv('FASTAPI_HOST',         '0.0.0.0')
FASTAPI_PORT      = int(os.getenv('FASTAPI_PORT',     '8000'))
MAX_FILE_SIZE_MB  = int(os.getenv('MAX_FILE_SIZE_MB','600'))

# Device
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_IN_8BIT      = torch.cuda.is_available()

WHISPER_MODEL_NAME = os.getenv('WHISPER_MODEL_NAME',   'openai/whisper-small')
ALIGN_BEAM_SIZE    = int(os.getenv('ALIGN_BEAM_SIZE',  '5'))