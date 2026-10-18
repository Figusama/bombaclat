"""Очередь задач анализа: один воркер (одна GPU/модель), статусы и результаты на диске."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
UPLOADS_DIR = ROOT / "data" / "uploads"


class JobManager:
    def __init__(self, analyzer_factory):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._factory = analyzer_factory
        self.analyzer = None
        self.ready = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        # восстановить завершённые задачи с диска
        for p in JOBS_DIR.glob("*.json"):
            try:
                d = json.load(open(p))
                self.jobs[d["job_id"]] = d
            except Exception:
                pass

    def _run(self):
        try:
            self.analyzer = self._factory()
        except Exception as e:
            log.exception("analyzer init failed: %s", e)
        self.ready.set()
        while True:
            job_id = self.q.get()
            job = self.jobs.get(job_id)
            if not job:
                continue
            job["status"] = "processing"
            job["started_at"] = time.time()
            try:
                def progress(stage, frac):
                    job["stage"] = stage
                    job["progress"] = round(frac, 2)
                if self.analyzer is None:
                    raise RuntimeError("analyzer is not available")
                res = self.analyzer.analyze(job["path"], progress=progress)
                job.update({"status": "done", "verdict": res["verdict"], "violations": res["violations"],
                            "meta": res["meta"], "progress": 1.0, "stage": "done"})
            except Exception as e:
                log.exception("job %s failed", job_id)
                job.update({"status": "failed", "error": str(e)})
            job["finished_at"] = time.time()
            self._persist(job)

    def _persist(self, job):
        try:
            json.dump(job, open(JOBS_DIR / f"{job['job_id']}.json", "w"), ensure_ascii=False)
        except Exception as e:
            log.warning("persist failed: %s", e)

    def submit(self, path: str, filename: str) -> str:
        job_id = uuid.uuid4().hex[:8]
        job = {"job_id": job_id, "status": "processing", "stage": "queued", "progress": 0.0,
               "filename": filename, "path": path, "created_at": time.time(),
               "queue_position": self.q.qsize()}
        # кеш готового результата: то же видео (по хешу файла) -> мгновенный ответ из предыдущей задачи
        try:
            from pipeline.facts import video_hash
            job["video_hash"] = video_hash(path)
            with self.lock:
                prev = next((j for j in sorted(self.jobs.values(), key=lambda j: -j.get("finished_at", 0))
                             if j.get("status") == "done" and j.get("video_hash") == job["video_hash"]), None)
            if prev:
                job.update({"status": "done", "stage": "done", "progress": 1.0, "verdict": prev["verdict"],
                            "violations": prev["violations"], "meta": {**prev.get("meta", {}), "cached_from": prev["job_id"]},
                            "finished_at": time.time()})
                with self.lock:
                    self.jobs[job_id] = job
                self._persist(job)
                log.info("job %s served from cache (same video as %s)", job_id, prev["job_id"])
                return job_id
        except Exception as e:
            log.warning("result cache lookup failed: %s", e)
        with self.lock:
            self.jobs[job_id] = job
        self.q.put(job_id)
        return job_id

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def public(self, job: dict) -> dict:
        out = {"job_id": job["job_id"], "status": job["status"]}
        if job["status"] == "processing":
            out.update({"stage": job.get("stage"), "progress": job.get("progress", 0.0)})
        elif job["status"] == "done":
            out.update({"verdict": job["verdict"], "violations": job["violations"], "meta": job.get("meta")})
        elif job["status"] == "failed":
            out["error"] = job.get("error")
        return out
