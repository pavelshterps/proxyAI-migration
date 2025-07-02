# config/celery.py

from config.settings import settings

# Брокер и бэкенд
broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

# Сериализация
task_serializer = "json"
accept_content = ["json"]

# Таймзона
timezone = settings.CELERY_TIMEZONE
enable_utc = True

# Здесь можно объявить периодические задачи, если нужно
beat_schedule = {
    # "some-task": {
    #     "task": "tasks.process_audio",
    #     "schedule": crontab(minute="*/5"),
    #     "args": (),
    # },
}