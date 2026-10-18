#!/usr/bin/env bash
# Поднимает cloudflared quick tunnel к локальному сервису и печатает публичный URL.
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a
PORT=${PORT:-8000}
pkill -f "cloudflared tunnel" 2>/dev/null
nohup cloudflared tunnel --url "http://localhost:$PORT" > data/tunnel.log 2>&1 &
for i in $(seq 1 30); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" data/tunnel.log | head -1)
  [ -n "$URL" ] && break
  sleep 1
done
echo "PUBLIC_URL=$URL"
echo "API_KEY=$API_KEY"
