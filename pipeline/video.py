"""Декодирование видео и сэмплинг кадров (PyAV)."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import av
import numpy as np
from PIL import Image


@dataclass
class Frame:
    t: float            # секунда от начала
    image: Image.Image  # RGB, оригинальное разрешение
    brightness: float   # средняя яркость (0..255)


@dataclass
class VideoFrames:
    path: str
    duration: float
    width: int
    height: int
    src_fps: float
    fps: float
    frames: list[Frame] = field(default_factory=list)

    def at(self, t: float) -> Frame:
        """Ближайший сэмплированный кадр к секунде t."""
        return min(self.frames, key=lambda f: abs(f.t - t))

    def between(self, t0: float, t1: float) -> list[Frame]:
        return [f for f in self.frames if t0 <= f.t < t1]


def probe(path: str) -> dict:
    with av.open(path) as c:
        s = c.streams.video[0]
        dur = float(c.duration / av.time_base) if c.duration else float(s.duration * s.time_base)
        return {"duration": dur, "width": s.codec_context.width, "height": s.codec_context.height,
                "fps": float(s.average_rate or 30)}


def sample_frames(path: str, fps: float = 2.0, max_side: int | None = None) -> VideoFrames:
    """Равномерный сэмплинг кадров с шагом 1/fps секунд. Кадры не ресайзятся, если max_side=None."""
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        src_fps = float(s.average_rate or 30)
        dur = float(c.duration / av.time_base) if c.duration else float(s.duration * s.time_base)
        vf = VideoFrames(path=path, duration=dur, width=s.codec_context.width,
                         height=s.codec_context.height, src_fps=src_fps, fps=fps)
        step = 1.0 / fps
        next_t = 0.0
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base) if fr.pts is not None else fr.time
            if t + 1e-6 < next_t:
                continue
            img = fr.to_image()
            if max_side and max(img.size) > max_side:
                k = max_side / max(img.size)
                img = img.resize((round(img.width * k), round(img.height * k)), Image.BILINEAR)
            gray = np.asarray(img.convert("L"), dtype=np.float32)
            vf.frames.append(Frame(t=round(t, 2), image=img, brightness=float(gray.mean())))
            next_t += step
    return vf


def to_jpeg_b64(img: Image.Image, quality: int = 80, max_side: int | None = None) -> str:
    if max_side and max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.BILINEAR)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def crop_box(img: Image.Image, box: tuple[float, float, float, float], pad: float = 0.15,
             upscale: float = 2.0, min_size: int = 160) -> Image.Image:
    """Кроп по нормализованному bbox (x0,y0,x1,y1 в 0..1) с полями и увеличением."""
    w, h = img.size
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0), (y1 - y0)
    x0 = max(0.0, x0 - bw * pad); x1 = min(1.0, x1 + bw * pad)
    y0 = max(0.0, y0 - bh * pad); y1 = min(1.0, y1 + bh * pad)
    c = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    if upscale != 1.0 or min(c.size) < min_size:
        k = max(upscale, min_size / max(1, min(c.size)))
        c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.BICUBIC)
    return c


def contact_sheet(frames: list[Frame], cols: int = 3, cell_w: int = 320, label: bool = True) -> Image.Image:
    """Сетка кадров с подписью времени (для VLM и отладки)."""
    from PIL import ImageDraw
    if not frames:
        raise ValueError("no frames")
    ratio = frames[0].image.height / frames[0].image.width
    cell_h = round(cell_w * ratio)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        im = f.image.resize((cell_w, cell_h), Image.BILINEAR)
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im, (x, y))
        if label:
            txt = f"t={f.t:.1f}s"
            d.rectangle((x, y, x + 8 + 7 * len(txt), y + 16), fill=(0, 0, 0))
            d.text((x + 4, y + 2), txt, fill=(255, 255, 0))
    return sheet


def annotate_frame(img: Image.Image, roi=None, label: str | None = None, t: float | None = None) -> Image.Image:
    """Кадр-доказательство: рамка вокруг зоны события + подпись времени."""
    from PIL import ImageDraw
    im = img.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    if roi:
        x0, y0, x1, y1 = roi
        pad_x, pad_y = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        box = (max(0, (x0 - pad_x) * w), max(0, (y0 - pad_y) * h), min(w, (x1 + pad_x) * w), min(h, (y1 + pad_y) * h))
        d.rectangle(box, outline=(255, 40, 40), width=4)
        if label:
            d.rectangle((box[0], max(0, box[1] - 18), box[0] + 8 + 7 * len(label), max(0, box[1] - 18) + 18), fill=(255, 40, 40))
            d.text((box[0] + 4, max(0, box[1] - 16)), label, fill=(255, 255, 255))
    if t is not None:
        txt = f"t={t:.1f}s"
        d.rectangle((0, 0, 10 + 7 * len(txt), 16), fill=(0, 0, 0))
        d.text((4, 2), txt, fill=(255, 255, 0))
    return im
