import os
import json
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

# Асинхронный движок и фабрика сессий
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

_whisper_model: WhisperModel | None = None
_diarizer: DiarizationPipeline | None = None

@signals.worker_process_init.connect
def preload_and_warmup(**kwargs):
    """При старте каждого worker’а:
       1) грузим пайплайн диаризации на CPU
       2) грузим quantized-Whisper на GPU из локального кэша (/hf_cache)"""
    global _diarizer, _whisper_model

    # 1) DiarizationPipeline
    try:
        _diarizer = DiarizationPipeline.from_pretrained(settings.PYANNOTE_PIPELINE)
        logger.info(f"Diarizer loaded: {settings.PYANNOTE_PIPELINE}")
    except Exception as e:
        logger.error(f"Cannot load diarization pipeline: {e}")
        raise

    # 2) WhisperModel — используем уже скачанный quantized-модельный каталог
    model_path = settings.WHISPER_MODEL_PATH
    whisper_kwargs = {
        "device": settings.WHISPER_DEVICE,
        "compute_type": settings.WHISPER_COMPUTE_TYPE,
        "batch_size": settings.WHISPER_BATCH_SIZE
    }
    if settings.HUGGINGFACE_CACHE_DIR:
        whisper_kwargs["cache_dir"] = settings.HUGGINGFACE_CACHE_DIR

    try:
        _whisper_model = WhisperModel(model_path, **whisper_kwargs)
        logger.info(f"Whisper model loaded from local path: {model_path}")
    except Exception as e:
        logger.error(f"Cannot load Whisper model: {e}")
        raise

@app.task(
    bind=True,
    name="process_audio",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
async def process_audio(self, upload_id: str, file_path: str):
    """Полный конвейер: диаризация → нарезка сегментов → транскрипция → JSON → статус"""
    db: AsyncSession = AsyncSessionLocal()
    try:
        # отметим начало обработки
        await update_upload_status(db, upload_id, "processing")

        # 1) Диаризация
        diar = _diarizer({"uri": file_path, "audio": file_path})
        segments = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        # 2) Транскрипция каждого сегмента
        transcripts = []
        for seg in segments:
            whisper_res = _whisper_model.transcribe(
                file_path,
                language=settings.WHISPER_LANGUAGE,
                word_timestamps=True,
                segments=[(seg["start"], seg["end"])]
            )
            # faster-whisper возвращает список сегментов
            words = []
            text = ""
            for part in whisper_res:
                text += part.text + " "
                words.extend(getattr(part, "words", []))
            transcripts.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": text.strip(),
                "words": words
            })

        # 3) Сохраняем в JSON
        os.makedirs(settings.RESULTS_FOLDER, exist_ok=True)
        output_path = os.path.join(settings.RESULTS_FOLDER, f"{upload_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2)

        # 4) Финальный статус + очистка исходников
        await update_upload_status(db, upload_id, "completed")
        cleanup_files(file_path)

    except Exception as e:
        logger.exception(f"Error processing upload {upload_id}: {e}")
        await update_upload_status(db, upload_id, "failed")
        raise
    finally:
        await db.close()

def cleanup_files(*paths: str):
    """Безопасно удаляем файлы после обработки"""
    for p in paths:
        try:
            os.remove(p)
            logger.info(f"Deleted file: {p}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {p}")
        except Exception as e:
            logger.warning(f"Could not delete {p}: {e}")