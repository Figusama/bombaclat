#!/usr/bin/env bash
# Запуск сервиса (локально) и, опционально, публичного туннеля cloudflared.
set -e
cd "$(dirname "$0")"
set -a; [ -f .env ] && source .env; set +a
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
PORT=${PORT:-8000}
if [ "$1" = "--tunnel" ]; then
  (cloudflared tunnel --url "http://localhost:$PORT" 2>&1 | tee data/tunnel.log) &
fi
exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
