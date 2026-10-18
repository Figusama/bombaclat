"""Калибровка сцены: поиск ROI (стол, сейф, ноутбук, дверь, счётная машинка) grounding'ом VLM.
Координаты не зависят от ракурса: модель находит объекты на первом кадре каждого видео."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from PIL import Image

from .vlm.base import VLMBackend, extract_json

log = logging.getLogger(__name__)

ROI_NAMES = {
    "desk": "the cashier's wooden desk (table)",
    "safe": "the metal safe (grey cabinet with a lock, usually standing on a pedestal)",
    "laptop": "the laptop computer",
    "door": "the entrance door of the room (glass door in the glass wall)",
    "counting_machine": "the banknote counting machine on the desk",
    "cashier_chair": "the office chair of the cashier",
}

GROUND_PROMPT = (
    "This is a surveillance frame of a bank cash office (one room with a desk). "
    "Locate the following objects and output their bounding boxes.\n"
    + "\n".join(f"- {k}: {v}" for k, v in ROI_NAMES.items())
    + "\nAnswer with JSON only, a list of objects: "
      '[{"label": "<name from the list>", "bbox_2d": [x1, y1, x2, y2]}]. '
      "Coordinates are integers normalized to 0-1000 for both axes (x: 0=left, 1000=right; y: 0=top, 1000=bottom). "
      "Output each label at most once (the largest / most clearly visible instance). Include only objects that are visible."
)


@dataclass
class Scene:
    width: int
    height: int
    rois: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)  # нормализованные xyxy

    def has(self, name: str) -> bool:
        return name in self.rois

    def room_mask(self):
        """Функция box->bool: человек внутри кассы (а не за стеклом).
        Эвристика: ноги человека ниже нижней кромки двери, либо бокс заметно пересекает стол."""
        door = self.rois.get("door")
        zones = [self.rois[k] for k in ("desk", "safe", "cashier_chair", "counting_machine") if k in self.rois]

        def inside(box):
            x0, y0, x1, y1 = box
            area = max(1e-6, (x1 - x0) * (y1 - y0))
            for z in zones:
                ix = max(0.0, min(x1, z[2]) - max(x0, z[0]))
                iy = max(0.0, min(y1, z[3]) - max(y0, z[1]))
                if ix * iy > 0.12 * area:
                    return True
            # камера смотрит из кассы на стеклянную стену: люди за стеклом всегда «выше» стола в кадре,
            # а у людей внутри низ бокса не выше верхней кромки стола
            desk = self.rois.get("desk")
            if desk:
                return y1 >= desk[1] - 0.03
            if door:
                return y1 >= door[3] - 0.03
            return y1 >= 0.35
        return inside

    def to_dict(self):
        return {"width": self.width, "height": self.height, "rois": {k: list(v) for k, v in self.rois.items()}}

    @classmethod
    def from_dict(cls, d):
        return cls(d["width"], d["height"], {k: tuple(v) for k, v in d["rois"].items()})


def _norm_box(b, w, h, coord_space: str):
    x0, y0, x1, y1 = [float(v) for v in b]
    if coord_space == "1000":
        x0, x1 = x0 / 1000, x1 / 1000
        y0, y1 = y0 / 1000, y1 / 1000
    else:
        x0, x1 = x0 / w, x1 / w
        y0, y1 = y0 / h, y1 / h
    x0, x1 = sorted((max(0, min(1, x0)), max(0, min(1, x1))))
    y0, y1 = sorted((max(0, min(1, y0)), max(0, min(1, y1))))
    return (x0, y0, x1, y1)


def calibrate(vlm: VLMBackend, frame: Image.Image, coord_space: str = "1000") -> Scene:
    """coord_space: 'pixels' (Qwen3-VL/3.5 abs pixels of input), '1000' (нормализованные 0..1000), 'auto'."""
    w, h = frame.size
    text = vlm.ask([frame], GROUND_PROMPT, max_tokens=500)
    obj = extract_json(text)
    scene = Scene(w, h)
    items = obj if isinstance(obj, list) else (obj.get("objects") if isinstance(obj, dict) else None) or []
    if not items:
        # запасной парсер: label ... [x,y,x,y]
        for m in re.finditer(r'"?(\w+)"?[^\[]*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text):
            items.append({"label": m.group(1), "bbox_2d": [int(m.group(i)) for i in range(2, 6)]})
    space = coord_space
    if space == "auto":
        mx = max([max(it.get("bbox_2d", [0])) for it in items] or [0])
        space = "1000" if mx > max(w, h) * 1.05 else "pixels"
    for it in items:
        label = str(it.get("label", "")).strip().lower().replace(" ", "_")
        box = it.get("bbox_2d") or it.get("bbox")
        if label in ROI_NAMES and box and len(box) == 4 and label not in scene.rois:
            scene.rois[label] = _norm_box(box, w, h, space)
    log.info("scene rois: %s", {k: [round(v, 2) for v in b] for k, b in scene.rois.items()})
    return scene


def draw_scene(frame: Image.Image, scene: Scene) -> Image.Image:
    from PIL import ImageDraw
    im = frame.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    for k, (x0, y0, x1, y1) in scene.rois.items():
        d.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=(255, 0, 0), width=2)
        d.text((x0 * w + 2, y0 * h + 2), k, fill=(255, 255, 0))
    return im
