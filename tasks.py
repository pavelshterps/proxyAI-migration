import os
import json
import logging

from celery import Celery, signals
from celery.utils.log import get_task_logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline

from config.settings import settings
from crud import update_upload_status

logger = get_task_logger(__name__)

app = Celery("proxyai")
app.config_from_object("config.celery")

# Асинхронный движок и сессии
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

_whisper: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None


@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    global _whisper, _diarizer

    # 1) Инициализация пайплайна диаризации на CPU
    try:
        # Если для gated pipeline нужен токен, можно добавить use_auth_token=settings.HF_TOKEN
        _diarizer = DiarizationPipeline.from_pretrained(settings.PYANNOTE_PIPELINE)
        logger.info(f"✅ Loaded diarization pipeline `{settings.PYANNOTE_PIPELINE}`")
    except Exception as e:
        logger.error(f"❌ Failed to load diarization pipeline: {e}")
        raise

    # 2) Предзагрузка Whisper (модель уже скачана и заквантована в float16)
    model_path = settings.WHISPER_MODEL_PATH

    # Общие параметры для CTranslate2
    whisper_kwargs = {
        "device": settings.WHISPER_DEVICE,                              # "cuda" или "cpu"
        "device_index": 0,                                              # для GPU
        "compute_type": settings.WHISPER_COMPUTE_TYPE,                  # "float16"
        "inter_threads": settings.GPU_CONCURRENCY if settings.WHISPER_DEVICE.startswith("cuda") else 1,
        "intra_threads": settings.CPU_CONCURRENCY if settings.WHISPER_DEVICE == "cpu" else 0,
        "flash_attention": False,
        "tensor_parallel": False,
    }

    # Пытаемся загрузить на GPU, иначе плавно откатываемся на CPU
    if settings.WHISPER_DEVICE.startswith("cuda"):
        try:
            _whisper = WhisperModel(model_path, **whisper_kwargs)
            logger.info(f"✅ Loaded Whisper model from `{model_path}` on GPU")
        except RuntimeError as gpu_err:
            logger.warning(f"❌ GPU load failed ({gpu_err}); falling back to CPU")
            cpu_kwargs = whisper_kwargs.copy()
            cpu_kwargs.update({
                "device": "cpu",
                "compute_type": "default",    # автоподбор оптимального для CPU
            })
            cpu_kwargs.pop("device_index", None)
            # для CPU не нужны inter_threads/intra_threads — они будут браться из настроек по умолчанию
            _whisper = WhisperModel(model_path, **cpu_kwargs)
            logger.info(f"✅ Loaded Whisper model from `{model_path}` on CPU fallback")
    else:
        _whisper = WhisperModel(model_path, **whisper_kwargs)
        logger.info(f"✅ Loaded Whisper model from `{model_path}` on CPU")


@app.task(
    bind=True,
    name="process_audio",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
async def process_audio(self, upload_id: int, file_path: str):
    session = AsyncSessionLocal()
    try:
        # Переводим запись в состояние processing
        await update_upload_status(session, upload_id, "processing")

        # 1) Диаризация
        diarization = _diarizer({"uri": file_path, "audio": file_path})
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # 2) Транскрипция каждого сегмента
        transcriptions = []
        for seg in segments:
            result = _whisper.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=False,
                segment=seg
            )
            text = " ".join([s.text for s in result])
            transcriptions.append({**seg, "text": text})

        # 3) Сохраняем результат в JSON
        json_path = f"{file_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

        # 4) Завершаем задачу
        await update_upload_status(session, upload_id, "completed")
        cleanup_files(file_path, json_path)

    except Exception as e:
        logger.exception(f"Error in process_audio (upload_id={upload_id}): {e}")
        await update_upload_status(session, upload_id, "failed")
        raise
    finally:
        await session.close()


def cleanup_files(*paths: str):
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"Deleted file {p}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"Failed to delete {p}: {e}")