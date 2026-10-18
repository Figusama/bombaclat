"""FastAPI-сервис: UI + API (X-API-Key) для анализа видео кассы."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.jobs import UPLOADS_DIR, JobManager  # noqa: E402
from pipeline.catalog import load_catalog  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

API_KEY = os.environ.get("API_KEY", "")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 500))

app = FastAPI(title="Cash Office Video Audit", version="1.0")
STATIC = ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _factory():
    from pipeline.analyze import Analyzer
    return Analyzer()


jobs = JobManager(_factory)


def require_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY is not configured on the server")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return True


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "model_ready": jobs.ready.is_set() and jobs.analyzer is not None,
            "queue": jobs.q.qsize()}


@app.get("/api/v1/catalog")
def catalog():
    c = load_catalog()
    return {"types": [c["types"][i] for i in c["order"]], "documents": c["documents"]}


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), _=Depends(require_key)):
    name = file.filename or "video.mov"
    ext = Path(name).suffix.lower() or ".mov"
    if ext not in (".mov", ".mp4", ".m4v", ".avi", ".mkv"):
        raise HTTPException(status_code=400, detail="unsupported file type; expected .mov")
    dst = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    size = 0
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="file too large")
            f.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty file")
    job_id = jobs.submit(str(dst), name)
    log.info("job %s queued: %s (%.1f MB)", job_id, name, size / 1e6)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str, _=Depends(require_key)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(jobs.public(job))


# --- те же эндпоинты для UI (ключ передаётся из поля на странице тем же заголовком) ---
@app.get("/api/v1/jobs/{job_id}/frame/{idx}/{k}")
def frame(job_id: str, idx: int, k: int, _=Depends(require_key)):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404)
    try:
        b64 = job["violations"][idx]["evidence_frames"][k]["image_base64"]
    except (IndexError, KeyError):
        raise HTTPException(status_code=404)
    import base64
    from fastapi.responses import Response
    return Response(content=base64.b64decode(b64), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
