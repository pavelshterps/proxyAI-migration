# tasks.py

import torch
from celery_app import celery_app
from config.settings import (
    DEVICE,
    LOAD_IN_8BIT,
    WHISPER_MODEL_NAME,
    WHISPER_COMPUTE_TYPE,
    ALIGN_MODEL_NAME,
    ALIGN_BEAM_SIZE,
    HUGGINGFACE_TOKEN,
)
from transformers import (
    AutoTokenizer,
    WhisperForConditionalGeneration,
)
# (если вы используете аудиопрепроцессинг)
# from transformers import WhisperProcessor

@celery_app.task(name="tasks.transcribe_task")
def transcribe_task(filepath: str) -> dict:
    """
    1) Загружает модель Whisper (8-bit на GPU, иначе full-precision на CPU)
    2) Прогоняет аудиофайл через модель
    3) Возвращает текст
    """

    # Инициализация токенизатора и модели
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

    # Подготовка входа
    inputs = tokenizer(
        filepath,
        return_tensors="pt",
        sampling_rate=16000
    ).to(DEVICE)

    # Генерация транскрипции
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {
        "text": transcription,
        "merge_task_id": None
    }