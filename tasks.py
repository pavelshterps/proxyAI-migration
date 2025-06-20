# tasks.py

import torch
import librosa
from celery_app import celery_app
from config.settings import (
    DEVICE,
    LOAD_IN_8BIT,
    WHISPER_MODEL_NAME,
    WHISPER_COMPUTE_TYPE,
    HUGGINGFACE_TOKEN,
)
from transformers import WhisperProcessor, AutoTokenizer, WhisperForConditionalGeneration

# Инициализация процессора, токенизатора и модели при импорте модуля
processor = WhisperProcessor.from_pretrained(
    WHISPER_MODEL_NAME,
    use_auth_token=HUGGINGFACE_TOKEN
)
tokenizer = AutoTokenizer.from_pretrained(
    WHISPER_MODEL_NAME,
    use_auth_token=HUGGINGFACE_TOKEN
)
model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_MODEL_NAME,
    device_map="auto" if LOAD_IN_8BIT else None,
    load_in_8bit=LOAD_IN_8BIT,
    torch_dtype=getattr(torch, WHISPER_COMPUTE_TYPE),
    use_auth_token=HUGGINGFACE_TOKEN
)
model.to(DEVICE)

@celery_app.task(name="tasks.transcribe_task")
def transcribe_task(filepath: str) -> dict:
    """
    Принимает путь к аудиофайлу, прогоняет его через Whisper:
    - 8-битный режим на GPU (если доступно)
    - иначе full-precision на CPU
    Возвращает словарь:
      {
        "text": "<транскрипт>",
        "merge_task_id": None
      }
    """
    print(f"Starting transcription for {filepath}")

    # Шаг 1: загрузка и препроцессинг аудио
    audio, sr = librosa.load(filepath, sr=16000)
    print(f"Audio loaded: {len(audio)} samples at {sr} Hz")
    # Preprocess audio into model inputs
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )
    # Move all tensors to the correct device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Шаг 2: инференс
    print("Running model.generate")
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)

    # Декодируем результат
    transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Transcription completed: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")

    return {
        "text": transcription,
        "merge_task_id": None
    }