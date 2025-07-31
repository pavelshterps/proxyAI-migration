#!/bin/bash
set -euo pipefail

if [ -z "${1-}" ]; then
  echo "Usage: $0 <upload_id>"
  exit 1
fi

UPLOAD_ID="$1"
WORKER_SERVICE="worker_gpu_2"  # поменяй, если нужно другой воркер/сервис
APP_ARG="-A celery_app"

echo "1. Получаем активные задачи diarize_full по upload_id=${UPLOAD_ID}"
ACTIVE_JSON=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG inspect active -j || true)

if [[ -z "$ACTIVE_JSON" ]]; then
  echo "Не удалось получить список активных задач или нет ответа."
else
  echo "Активные задачи на всех нодах:"
  echo "$ACTIVE_JSON" | jq .
fi

# Находим все active diarize_full задачи с args[0] == upload_id
MATCHING_IDS=$(echo "$ACTIVE_JSON" | jq -r --arg UID "$UPLOAD_ID" '
  to_entries[]
  | .value.active? // []
  | .[]
  | select(.name == "tasks.diarize_full")
  | (try (.args | fromjson) catch .) as $args
  | select($args[0] == $UID)
  | {id: .id, time_start: .time_start, hostname: .hostname}
' | jq -s '.')

# Выбираем одну оставшуюся (самую "свежую" по time_start)
KEEP_ID=$(echo "$MATCHING_IDS" | jq -r 'sort_by(.time_start) | last | .id // empty')

if [[ -n "$KEEP_ID" ]]; then
  echo "2. Оставляем существующую задачу: $KEEP_ID"
else
  echo "2. Ни одной живой задачи не найдено, будет создана новая."
fi

# Отзываем остальные дубликаты (если больше одной)
TO_REVOKE=$(echo "$MATCHING_IDS" | jq -r --arg KEEP "$KEEP_ID" '
  .[] | select(.id != $KEEP) | .id
' | sort | uniq)

if [[ -n "$TO_REVOKE" ]]; then
  echo "3. Отзываем дубликаты:"
  for tid in $TO_REVOKE; do
    echo "  revoke $tid"
    docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG control revoke "$tid" || true
  done
else
  echo "3. Дубликатов для отзыва не найдено."
fi

# Если нет оставшейся, запускаем новую
if [[ -z "$KEEP_ID" ]]; then
  echo "4. Запускаем новую задачу diarize_full"
  NEW_ID=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG call tasks.diarize_full --args="[\"$UPLOAD_ID\", null]" --queue=diarize_gpu)
  echo "  Запрошена новая задача: $NEW_ID"
else
  echo "4. Пропускаем запуск, живая задача уже есть."
fi

echo "5. Текущее состояние active после ревокаций/создания:"
docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG inspect active -j | jq .

echo
echo "Готово. Следи логи воркера для прогресса:"
echo "  docker-compose logs -f $WORKER_SERVICE"