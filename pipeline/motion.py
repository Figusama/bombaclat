"""Пиксельные сигналы по зонам: активность в зоне счётной машинки (использование машинки) и т.п."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .video import VideoFrames


def _roi_gray(img: Image.Image, roi, size=(64, 48)) -> np.ndarray:
    w, h = img.size
    x0, y0, x1, y1 = roi
    c = img.crop((int(x0 * w), int(y0 * h), max(int(x0 * w) + 2, int(x1 * w)), max(int(y0 * h) + 2, int(y1 * h))))
    return np.asarray(c.convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def roi_activity(vf: VideoFrames, roi) -> list[float]:
    """Средняя абсолютная разность соседних кадров внутри ROI (по сэмплам vf)."""
    prev = None
    out = []
    for f in vf.frames:
        g = _roi_gray(f.image, roi)
        out.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
        prev = g
    return out


def active_intervals(vf: VideoFrames, roi, k: float = 3.0, min_abs: float = 6.0, min_len: float = 1.5) -> list[tuple[float, float]]:
    """Интервалы, где активность в ROI заметно выше фоновой: > k * медиана и > min_abs, длительностью >= min_len."""
    act = roi_activity(vf, roi)
    if len(act) < 4:
        return []
    base = float(np.median(act))
    thr = max(min_abs, k * base)
    ts = [f.t for f in vf.frames]
    out = []
    start = None
    for t, a in zip(ts, act):
        if a >= thr and start is None:
            start = t
        elif a < thr and start is not None:
            if t - start >= min_len:
                out.append((start, t))
            start = None
    if start is not None and ts[-1] + 0.5 - start >= min_len:
        out.append((start, ts[-1] + 0.5))
    return out
