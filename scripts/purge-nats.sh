#!/usr/bin/env bash
set -euo pipefail

CONTAINER=${1:-nats-js}
STREAM=${2:-SNN_EVENTS}

docker exec "$CONTAINER" /bin/sh -c '
  set -e
  if ! command -v nats >/dev/null 2>&1; then
    apk add --no-cache natscli >/dev/null 2>&1
  fi
  nats stream purge "'"$STREAM"'" --force
'

echo "Purged stream '$STREAM' in container '$CONTAINER'."
