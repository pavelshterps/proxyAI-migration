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
    WhisperProcessor,
)
import librosa

@celery_app.task(name="tasks.transcribe_task")
def transcribe_task(filepath: str) -> dict:
    """
    Принимает путь к аудиофайлу, прогоняет через Whisper (8-bit на GPU или full-precision на CPU),
    возвращает результат в формате:
      {
        "text": "...",
        "merge_task_id": None
      }
    """

    # 1) Загрузка модели
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

    # 2) Инициализация процессора
    processor = WhisperProcessor.from_pretrained(
        WHISPER_MODEL_NAME,
        use_auth_token=HUGGINGFACE_TOKEN
    )

    # 3) Загрузка и препроцессинг аудио
    audio, sr = librosa.load(filepath, sr=16000)
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    ).to(DEVICE)

    # 4) Генерация транскрипта
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {
        "text": transcription,
        "merge_task_id": None
    }