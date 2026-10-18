# Данные для жюри

- Публичный URL (UI): https://traditional-carey-workers-apparatus.trycloudflare.com
- API base: https://traditional-carey-workers-apparatus.trycloudflare.com/api/v1
- API-ключ (заголовок `X-API-Key`): передаётся организаторам отдельно (значение `API_KEY` из `.env` на сервере)

Пример:
```bash
curl -X POST "$URL/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@video.mov"
curl "$URL/api/v1/jobs/<job_id>" -H "X-API-Key: $KEY"
```

Сервис работает на ноутбуке команды (Apple M4 Pro), один воркер; видео обрабатываются по очереди,
одно видео — от 10 с до ~3 мин в зависимости от длины.
