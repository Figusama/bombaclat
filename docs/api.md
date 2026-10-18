# API

Все запросы требуют заголовок `X-API-Key: <ключ>` (значение `API_KEY` из `.env`). Без ключа — `401`.

## POST /api/v1/analyze

Загрузка видео (`multipart/form-data`, поле `file`, формат `.mov`).

```bash
curl -X POST https://<host>/api/v1/analyze \
     -H "X-API-Key: <ключ>" -F "file=@video_017.mov"
# {"job_id": "3f2a9c1e", "status": "processing"}
```

## GET /api/v1/jobs/{job_id}

Статус (`processing` | `done` | `failed`) и результат.

```bash
curl -H "X-API-Key: <ключ>" https://<host>/api/v1/jobs/3f2a9c1e
```

Ответ при `processing`:
```json
{"job_id": "3f2a9c1e", "status": "processing", "stage": "facts", "progress": 0.45}
```

Ответ при `done` (поля по ТЗ + дополнительные):
```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Уход: ключ оставлен в сейфе",
      "type_id": "safe_key_left",
      "timecode_sec": 14.0,
      "description": "Кассир покинул кассу, а ключ остался в замке сейфа.",
      "evidence_frames": [{"timestamp_sec": 14.0, "image_base64": "/9j/4AAQ..."}],
      "priority": "High",
      "confidence": 0.7,
      "rule": {"document": "Постановление Правления НБРК от 29.11.2019 № 231 ...", "clause": "п. 80",
               "url": "https://adilet.zan.kz/rus/docs/V1900019680", "quote": "...", "basis": "confirmed"},
      "rules_all": [...],
      "observed": {"empty_from": 13.5}
    }
  ],
  "meta": {"duration_sec": 21.0, "timings": {"total": 48.3}, "segments": [...], "people_timeline": {...}}
}
```

`verdict` ∈ {`violation`, `no_violation`}; `violations` пуст, если нарушений нет. `evidence_frames` — 1–3 JPEG в base64.

## GET /api/v1/health — без ключа; готовность модели и длина очереди.
## GET /api/v1/catalog — каталог типов нарушений с нормативными ссылками.
