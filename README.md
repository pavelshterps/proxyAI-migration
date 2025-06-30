
# proxyAI

**Version:** 13.7.6.1  
**License:** MIT

**proxyAI** — это микросервис для speaker-diarization и транскрипции аудио на GPU с разбиением на чанки по голосу, основанный на FastAPI + Celery + Faster-Whisper + PyAnnote.

---

## Содержание

1. [Архитектура](#архитектура)  
2. [Требования](#требования)  
3. [Установка и локальный запуск](#установка-и-локальный-запуск)  
4. [Docker и Docker Compose](#docker-и-docker-compose)  
5. [Конфигурация (`.env`)](#конфигурация-env)  
6. [API-эндпоинты](#api-эндпоинты)  
7. [Фоновые задачи (Celery)](#фоновые-задачи-celery)  
8. [Warm-up моделей](#warm-up-моделей)  
9. [Логирование](#логирование)  
10. [Метрики и мониторинг](#метрики-и-мониторинг)  
11. [Тестирование](#тестирование)  
12. [CI/CD](#cicd)  
13. [Отладка и распространённые ошибки](#отладка-и-распространённые-ошибки)  
14. [Дальнейшие улучшения](#дальнейшие-улучшения)  

---

## Архитектура

```
┌─────────────────────────┐
│       FastAPI API       │
│  /upload, /transcribe,  │
│   /results, /health     │
└───┬─────────────────┬───┘
    │                 │
    ▼                 ▼
Celery Producer    Metrics + Logs
    │
    ▼
┌───────────────┐   ┌────────────────┐
│  CPU Worker   │   │  GPU Worker    │
│ (diarization) │   │(transcription) │
└───────────────┘   └────────────────┘
      │                   │
      ▼                   ▼
  Файловая система: /data/uploads, /data/results
      │
      └─► Cleaner (Celery Beat ежедневно)
```

---

## Требования

- **Python 3.10+**  
- **Docker 20.10+** & **Docker Compose 1.29+**  
- **NVIDIA GPU** + драйверы + nvidia-container-runtime  
- **Redis** (брокер Celery)  
- **HuggingFace-токен** для PyAnnote  

---

## Установка и локальный запуск

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/pavelshterps/proxyAI.git
   cd proxyAI
   ```
2. Создать и активировать виртуальное окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Скопировать пример конфигурации:
   ```bash
   cp .env.example .env
   ```
   и заполнить свои значения.
4. Создать необходимые папки и загрузить модель:
   ```bash
   mkdir -p /data/uploads /data/results /data/diarizer_cache
   mkdir -p /hf_cache/models--guillaumekln--faster-whisper-medium
   ```
5. Запустить Redis (если локально):
   ```bash
   redis-server --daemonize yes
   ```
6. Запустить API:
   ```bash
   export API_WORKERS=1
   gunicorn main:app -k uvicorn.workers.UvicornWorker -w $API_WORKERS -b 0.0.0.0:8000
   ```
7. Запустить Celery Workers:
   ```bash
   # CPU
   celery -A celery_app worker --loglevel=info --concurrency=$CPU_CONCURRENCY --queues=preprocess_cpu

   # GPU (nvidia runtime)
   celery -A celery_app worker --loglevel=info --concurrency=1 --queues=preprocess_gpu
   ```
8. (Опционально) Запустить Beat:
   ```bash
   celery -A celery_app beat --loglevel=info
   ```
9. (Опционально) Запустить metrics-сервер:
   ```bash
   python metrics.py
   ```

---

## Docker и Docker Compose

```bash
docker-compose up -d --build
```

Сервисная сетка:
- **api** — FastAPI + Gunicorn  
- **cpu-worker** — Celery CPU  
- **gpu-worker** — Celery GPU  
- **beat** — Celery Beat  
- **metrics** — Prometheus-клиент  
- **dcgm-exporter** — NVIDIA DCGM  
- **redis** — Redis  
- **flower** — Flower UI  

Проверить статусы:
```bash
docker-compose ps
```

---

## Конфигурация (`.env`)

```dotenv
# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
CELERY_TIMEZONE=UTC

# Concurrency
API_WORKERS=1
CPU_CONCURRENCY=1
GPU_CONCURRENCY=1

# Paths
UPLOAD_FOLDER=/data/uploads
RESULTS_FOLDER=/data/results
DIARIZER_CACHE_DIR=/data/diarizer_cache

# Models
WHISPER_MODEL_PATH=/hf_cache/models--guillaumekln--faster-whisper-medium
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8
PYANNOTE_PROTOCOL=pyannote/speaker-diarization
HUGGINGFACE_TOKEN=<Ваш_Token>

# Segmentation / VAD
SEGMENT_LENGTH_S=30
VAD_LEVEL=2

# File limits & retention
MAX_FILE_SIZE=1073741824
FILE_RETENTION_DAYS=7

# Tus endpoint
TUS_ENDPOINT=http://tus.example.com/files/

# Frontend / CORS
ALLOWED_ORIGINS=http://localhost:3000

# Metrics exporter
METRICS_PORT=8001

# Flower UI auth
FLOWER_USER=admin
FLOWER_PASS=secret
```

---

## API-эндпоинты

### `POST /upload/`
- **Описание**: загрузка аудио и запуск задач  
- **Заголовки**: `X-Correlation-ID` (опционально)  
- **Форма**: `file: UploadFile` (`audio/wav` | `audio/mpeg`)  
- **Ответ 200**:
  ```json
  HTTP/1.1 200 OK
  X-Correlation-ID: <cid>
  {
    "upload_id": "<filename_or_uuid>"
  }
  ```
- **Ошибки**: `415`, `413`  
- **Rate-limit**: 10 req/min per IP  

### `POST /transcribe`
Аналогично `/upload/`.  

### `GET /results/{upload_id}`
- **Описание**: отдаёт `transcript.json` и `diarization.json`  
- **Ответ 200**:
  ```json
  {
    "transcript": "[…]",
    "diarization": "[…]"
  }
  ```
- **Ошибки**: `404 Not Found`  
- **Rate-limit**: 20 req/min per IP  

### `GET /health`
- **Response**: `{"status":"ok","version":"13.7.6.1"}`  
- **Rate-limit**: 30 req/min  

### `GET /metrics`
- **Метрики Prometheus**:  
  - HTTP: `http_requests_total`, `http_request_duration_seconds`  
  - Celery: `vad_segmentation_seconds`, `whisper_transcription_seconds`, `celery_task_runs_total`  
- **Rate-limit**: 10 req/min  

---

## Фоновые задачи (Celery)

1. **`tasks.diarize_full(upload_id, correlation_id)`**  
   Полная diarization через PyAnnote.
2. **`tasks.transcribe_segments(upload_id, correlation_id)`**  
   VAD сегментация + Faster-Whisper транскрипция.
3. **`tasks.cleanup_old_files()`**  
   Удаление старых файлов по расписанию.

**Настройки**:  
- `acks_late=True`, `task_reject_on_worker_lost=True`  
- `worker_prefetch_multiplier=1`  
- `task_time_limit=600`, `task_soft_time_limit=550`  
- `autoretry_for` и `retry_backoff`

---

## Warm-up моделей

При старте каждого воркера выполняется «разогрев»:
```python
whisper.transcribe(sample.wav, offset=0, duration=2.0, language="ru", vad_filter=True)
diarizer(sample.wav)
```
— чтобы CUDA-контексты и кэши были готовы и избежать задержки первой реальной задачи.

---

## Логирование

- **StructLog** в JSON  
- Поля: `timestamp`, `level`, `event`, `upload_id`, `correlation_id`, `error` и др.

Пример:
```json
{
  "timestamp":"2025-07-01T12:00:00Z",
  "level":"info",
  "event":"transcription start",
  "upload_id":"sample.wav",
  "correlation_id":"123e4567-e89b-12d3-a456-426614174000"
}
```

---

## Метрики и мониторинг

- **Prometheus**:  
  - `/metrics` в API  
  - `metrics` сервис на `METRICS_PORT` с Celery-метриками  
- **GPU**:  
  - NVIDIA DCGM-Exporter на порту `9400`  

---

## Тестирование

- **Unit**: функции VAD, конфиг, зависимости  
- **Integration**: `tests/test_integration.py`  
- **Eager mode** (CI):  
  ```bash
  export CELERY_TASK_ALWAYS_EAGER=True
  pytest
  ```
- **Lint & Security**: `flake8`, `safety check`

---

## CI/CD

- **GitHub Actions**:  
  - `flake8 .`  
  - `pytest`  
  - `safety check`  
- **Coverage** > 80%

---

## Отладка и распространённые ошибки

- **PydanticImportError** → убедитесь, что `BaseSettings` импортируется из `pydantic_settings`.  
- **Extra inputs** → добавьте `Field(..., env="…")` и `extra="ignore"`.  
- **Flower image** → используйте `mher/flower:latest`.  
- **GPU OOM** → `concurrency=1`, `dcgm-exporter` запущен.

---


## Demo UI

A full single-page demo lives in `static/index.html`. It allows you to:

1. Enter your **X-API-Key** (get it via `POST /admin/users`).  
2. Upload an audio file to `POST /upload/`.  
3. Poll progress at `GET /status/{upload_id}`.  
4. Fetch results at `GET /results/{upload_id}`, view segments with **time codes** and **play buttons**.  
5. Edit speaker labels inline and save via `POST /labels/{upload_id}`.

## Running tests

```bash
pytest tests/test_labels_status.py
