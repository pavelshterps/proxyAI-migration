#!/bin/bash
set -euo pipefail

if [ -z "${1-}" ]; then
  echo "Usage: $0 <upload_id>"
  exit 1
fi

UPLOAD_ID="$1"
WORKER_SERVICE="worker_gpu_2"  # поменяй, если нужен другой
APP_ARG="-A celery_app"

echo "1. Получаем активные задачи diarize_full по upload_id=${UPLOAD_ID}"
ACTIVE_JSON=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG inspect active -j || true)

if [[ -z "$ACTIVE_JSON" ]]; then
  echo "Не удалось получить список активных задач или нет ответа."
else
  echo "Активные задачи на всех нодах:"
  echo "$ACTIVE_JSON" | jq .
fi

# Собираем все задачи diarize_full, у которых args[0] == upload_id
# Формат: node, id, time_start
MAP=$(echo "$ACTIVE_JSON" | jq -c --arg UID "$UPLOAD_ID" '
  to_entries
  | map(
      {node: .key, tasks: (.value.active? // [])}
    )
  | map(
      .tasks[]
      | select(.name == "tasks.diarize_full")
      | select((.args | type == "array" and .[0] == $UID))
      | {node: .hostname, id: .id, time_start: .time_start}
    )
')

# Если нет активных — будем запускать новую
if [[ -z "$MAP" || "$MAP" == "null" ]]; then
  echo "2. Дубликатов не найдено (активных diarize_full для этого upload_id нет)."
  echo "3. Вызываем новую задачу diarize_full"
  NEW_ID=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG call tasks.diarize_full --args="[\"$UPLOAD_ID\", null]" --queue=diarize_gpu)
  echo "  Запрошена новая задача: $NEW_ID"
else
  # Найдём задачу с минимальным time_start — её сохраняем, остальные ревоким
  echo "2. Найденные активные diarize_full задачи с args[0]=${UPLOAD_ID}:"
  echo "$MAP" | jq .

  KEEP_ID=$(echo "$MAP" | jq -r 'sort_by(.time_start)[0].id')
  echo "  Оставляем задачу: $KEEP_ID"

  # Ревоким все остальные
  TO_REVOKE=$(echo "$MAP" | jq -r --arg keep "$KEEP_ID" '. | map(select(.id != $keep)) | .[].id' | sort | uniq)
  if [[ -n "$TO_REVOKE" ]]; then
    echo "  Отзываем дубликаты:"
    for tid in $TO_REVOKE; do
      echo "    revoke $tid"
      docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG control revoke "$tid" || true
    done
  else
    echo "  Дубликатов для отзыва не найдено (уже одна активная)."
  fi

  echo "3. Проверяем: если ни одной оставшейся задачи нет (вдруг была убита), то запускаем новую"
  # Повторно выгрузим active и убедимся что KEEP_ID всё ещё активна
  UPDATED=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG inspect active -j || true)
  STILL_THERE=$(echo "$UPDATED" | jq -e --arg keep "$KEEP_ID" '
    to_entries
    | any(.value.active? // [] | .[] | select(.id == $keep))
  ' >/dev/null 2>&1 && echo "yes" || echo "no")

  if [[ "$STILL_THERE" != "yes" ]]; then
    echo "  Оставшаяся задача исчезла, запускаем новую diarize_full"
    NEW_ID=$(docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG call tasks.diarize_full --args="[\"$UPLOAD_ID\", null]" --queue=diarize_gpu)
    echo "    Запрошена новая задача: $NEW_ID"
  else
    echo "  Задача $KEEP_ID продолжает висеть — новую не запускаем."
  fi
fi

echo "4. Ждём секунду и показываем текущее состояние"
sleep 1
docker-compose exec -T "$WORKER_SERVICE" celery $APP_ARG inspect active -j | jq .

echo
echo "Готово. Следи логи воркера для прогресса:"
echo "  docker-compose logs -f $WORKER_SERVICE"