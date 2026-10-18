"""Детекция людей (RT-DETR, Apache-2.0) и временная шкала присутствия в кассе."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .video import VideoFrames

log = logging.getLogger(__name__)

MODEL_ID = "PekingU/rtdetr_r50vd"


@dataclass
class Det:
    box: tuple[float, float, float, float]  # нормализованный xyxy 0..1
    score: float
    label: str


@dataclass
class PersonTimeline:
    t: list[float]
    count: list[int]                    # сглаженное число людей в кадре
    raw_count: list[int]
    dets: list[list[Det]] = field(default_factory=list)

    def count_at(self, ts: float) -> int:
        i = int(np.argmin(np.abs(np.asarray(self.t) - ts)))
        return self.count[i]

    def segments(self) -> list[tuple[float, float, int]]:
        """Интервалы постоянного числа людей: (t0, t1, count)."""
        out = []
        if not self.t:
            return out
        t0, c0 = self.t[0], self.count[0]
        for i in range(1, len(self.t)):
            if self.count[i] != c0:
                out.append((t0, self.t[i], c0))
                t0, c0 = self.t[i], self.count[i]
        out.append((t0, self.t[-1] + 0.5, c0))
        return out

    def events(self) -> list[dict]:
        """Изменения числа людей: {'t','from','to'}."""
        ev = []
        for i in range(1, len(self.t)):
            if self.count[i] != self.count[i - 1]:
                ev.append({"t": self.t[i], "from": self.count[i - 1], "to": self.count[i]})
        return ev


class PersonDetector:
    def __init__(self, model_id: str = MODEL_ID, threshold: float = 0.5, device: str | None = None):
        from transformers import AutoImageProcessor, RTDetrForObjectDetection
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.id2label = self.model.config.id2label
        self.threshold = threshold

    @torch.no_grad()
    def detect(self, images: list[Image.Image], batch: int = 8) -> list[list[Det]]:
        out: list[list[Det]] = []
        for i in range(0, len(images), batch):
            ims = [im.convert("RGB") for im in images[i:i + batch]]
            inputs = self.processor(images=ims, return_tensors="pt").to(self.device)
            res = self.model(**inputs)
            sizes = torch.tensor([[im.height, im.width] for im in ims])
            post = self.processor.post_process_object_detection(res, threshold=self.threshold, target_sizes=sizes)
            for im, p in zip(ims, post):
                dets = []
                for s, l, b in zip(p["scores"], p["labels"], p["boxes"]):
                    x0, y0, x1, y1 = b.tolist()
                    dets.append(Det((x0 / im.width, y0 / im.height, x1 / im.width, y1 / im.height),
                                    float(s), self.id2label[int(l)]))
                out.append(dets)
        return out


def _iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def persons_in_room(dets: list[Det], room_mask=None, min_h: float = 0.08) -> list[Det]:
    """Люди внутри кассы. room_mask(box)->bool отсекает людей за стеклом (если известна зона)."""
    ps = [d for d in dets if d.label == "person" and (d.box[3] - d.box[1]) >= min_h]
    # дедуп
    keep: list[Det] = []
    for d in sorted(ps, key=lambda d: -d.score):
        if all(_iou(d.box, k.box) < 0.5 for k in keep):
            keep.append(d)
    if room_mask is not None:
        keep = [d for d in keep if room_mask(d.box)]
    return keep


def build_timeline(vf: VideoFrames, detector: PersonDetector, room_mask=None, smooth: int = 3) -> PersonTimeline:
    all_dets = detector.detect([f.image for f in vf.frames])
    raw = []
    per_frame = []
    for dets in all_dets:
        ps = persons_in_room(dets, room_mask)
        per_frame.append(ps)
        raw.append(len(ps))
    arr = np.asarray(raw)
    if smooth > 1 and len(arr) >= smooth:
        k = smooth // 2
        sm = [int(np.median(arr[max(0, i - k):i + k + 1])) for i in range(len(arr))]
    else:
        sm = raw
    return PersonTimeline(t=[f.t for f in vf.frames], count=sm, raw_count=raw, dets=per_frame)


def _overlap_frac(box, roi) -> float:
    ix = max(0.0, min(box[2], roi[2]) - max(box[0], roi[0]))
    iy = max(0.0, min(box[3], roi[3]) - max(box[1], roi[1]))
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return ix * iy / area


def desk_presence(tl: PersonTimeline, desk_roi, min_frac: float = 0.08, smooth: int = 3) -> list[bool]:
    """Есть ли в кадре человек у стола (бокс заметно пересекает ROI стола), по кадрам шкалы."""
    raw = [any(_overlap_frac(d.box, desk_roi) >= min_frac for d in dets) for dets in tl.dets]
    if smooth > 1 and len(raw) >= smooth:
        k = smooth // 2
        return [sum(raw[max(0, i - k):i + k + 1]) * 2 > len(raw[max(0, i - k):i + k + 1]) for i in range(len(raw))]
    return raw
