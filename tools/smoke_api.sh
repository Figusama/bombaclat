#!/usr/bin/env bash
# Сквозная проверка API: загрузка видео, опрос статуса, проверка схемы ответа и декодируемости кадров.
# Использование: tools/smoke_api.sh [URL] [VIDEO]   (ключ берётся из .env)
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a
URL=${1:-http://localhost:${PORT:-8000}}
VIDEO=${2:-data/open/videos/X-26.mov}
echo "# без ключа -> ожидаем 401"
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$URL/api/v1/analyze" -F "file=@$VIDEO"
echo "# с ключом"
JOB=$(curl -s -X POST "$URL/api/v1/analyze" -H "X-API-Key: $API_KEY" -F "file=@$VIDEO" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "job_id=$JOB"
while true; do
  R=$(curl -s "$URL/api/v1/jobs/$JOB" -H "X-API-Key: $API_KEY")
  S=$(echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], d.get('stage',''), d.get('progress',''))")
  echo "  $S"
  case "$S" in done*|failed*) break;; esac
  sleep 3
done
echo "$R" | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
print('verdict:', d['verdict'])
for v in d.get('violations', []):
    frames = v['evidence_frames']
    sizes = [len(base64.b64decode(f['image_base64'])) for f in frames]
    print(' -', v['type'], '@', v['timecode_sec'], '| frames:', [(f['timestamp_sec'], s) for f, s in zip(frames, sizes)])
    print('   ', v['description'][:100])
    print('   ', v['rule']['document'][:60], v['rule']['clause'])
print('total sec:', d['meta']['timings']['total'])
"
