#!/bin/bash
set -euo pipefail

if [ -z "${1-}" ]; then
  echo "Usage: $0 <upload_id>"
  exit 1
fi

UPLOAD_ID="$1"
WORKER_SERVICE="worker_gpu_2"  # поменяй если нужно другой
APP_ARG="-A celery_app"

echo "1. Получаем активные задачи diarize_full по upload_id=${UPLOAD_ID}"
ACTIVE_JSON=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG inspect active -j || true)

if [[ -z "$ACTIVE_JSON" ]]; then
  echo "Не удалось получить список активных задач или нет ответа."
else
  echo "Активные задачи на всех нодах:"
  echo "$ACTIVE_JSON" | jq .
fi

# Находим все идентификаторы задач diarize_full с аргументом upload_id
TO_REVOKE=$(echo "$ACTIVE_JSON" | jq -r --arg UID "$UPLOAD_ID" '
  to_entries[]
  | .value.active? // []
  | .[]
  | select(.name == "tasks.diarize_full")
  | select((.args | tostring) | test($UID))
  | .id
' | sort | uniq)

if [[ -n "$TO_REVOKE" ]]; then
  echo "2. Отзываем дубликаты (если есть):"
  for tid in $TO_REVOKE; do
    echo "  revoke $tid"
    docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG control revoke "$tid" || true
  done
else
  echo "2. Дубликатов не найдено."
fi

echo "3. Вызываем новую задачу diarize_full"
NEW_ID=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG call tasks.diarize_full --args="[\"$UPLOAD_ID\", null]" --queue=diarize_gpu)
echo "  Запрошена новая задача: $NEW_ID"

echo "4. Ждём секунду и показываем текущее состояние"
sleep 1
docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG inspect active -j | jq .

echo
echo "Готово. Следи логи воркера для прогресса:"
echo "  docker-compose logs -f $WORKER_SERVICE"