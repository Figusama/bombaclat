"""Извлечение наблюдаемых фактов из видео через VLM: сегментация, промпты, нормализация, кеш."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from .persons import PersonTimeline
from .scene import Scene
from .video import Frame, VideoFrames, crop_box
from .vlm.base import VLMBackend

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "facts"
PROMPT_VERSION = "v5"

ACTIONS = ["idle", "counting_machine", "counting_hand", "handing_cash", "receiving_cash",
           "checking_document", "at_safe", "writing_or_typing", "away", "other"]
SCREEN = ["work", "lock", "off", "closed", "unknown"]

SCENE_INTRO = (
    "You are analysing surveillance frames from a bank cash office: one small room with a desk, "
    "a laptop, a banknote counting machine, a metal safe, and a glass wall with an entrance door. "
    "People seen through the glass wall are OUTSIDE the room and must be ignored. "
    "The cashier is the bank employee who works at the desk; a client or cash collector may sit or stand "
    "on the opposite side of the desk. Report only what is clearly visible; never guess."
)

FACT_FIELDS = """
"people_count": <integer: people INSIDE the room, including the cashier>,
"cashier_at_desk": <true/false: the cashier is at the desk (sitting or standing right by it)>,
"client_at_desk": <true/false: a visitor (client or collector) is at the desk>,
"cashier_action": <one of: idle, counting_machine (feeding banknotes into the counting machine), counting_hand (counting or fanning banknotes with hands), handing_cash (giving banknotes to the visitor), receiving_cash (taking banknotes from the visitor), checking_document (holding/looking at an ID card or paper), at_safe (standing at the safe, opening/closing it), writing_or_typing, away (cashier not in the room), other>,
"money_on_desk": <true/false: banknotes lying on the desk surface>,
"bag_on_desk": <true/false: a cash collection bag or pouch on the desk>,
"personal_items_on_desk": <true/false: handbag, backpack, jacket or coat on the desk>,
"food_drink_on_desk": <true/false: cup, mug, bottle, snack or food on the desk>,
"phone_in_hand": <true/false: a person holds a smartphone>,
"document_on_desk": <true/false: an ID card, passport or paper document on the desk or in hands>,
"safe_door_open": <true/false: the safe door is open>,
"key_in_safe_lock": <true/false: a key is inserted in the safe lock (look closely at the safe door)>,
"laptop_screen": <one of: work (bright screen with windows/applications), lock (login or lock screen, mostly uniform wallpaper with clock/avatar), off (dark screen), closed (lid closed), unknown>,
"door_open": <true/false: the entrance door of the room is open or ajar>,
"lights_on": <true/false: room lights are on (bright room)>
""".strip()

FACT_PROMPT_SINGLE = (
    SCENE_INTRO + "\nLook at this single frame and fill the JSON below. Answer with JSON only.\n{\n"
    + FACT_FIELDS + "\n}"
)


def segment_prompt(ts: list[float], has_crops: bool) -> str:
    frames_desc = ", ".join(f"t={t:.1f}s" for t in ts)
    crops = ""
    if has_crops:
        crops = (" After the full frames there are zoomed crops (labelled 'safe', 'laptop', 'desk') "
                 "taken from the LAST frame; use them to answer about the safe, key, screen and desk items.")
    return (
        SCENE_INTRO
        + f"\nYou are given {len(ts)} consecutive frames ({frames_desc}); the time is printed on each frame."
        + crops
        + "\nFill the JSON below. Fields marked 'any' are true if the thing is visible in ANY frame; "
          "'end' fields describe the LAST frame. Answer with JSON only.\n{\n"
          '"people_count_max": <integer: max people INSIDE the room in any frame>,\n'
          '"people_count_end": <integer: people INSIDE the room in the last frame>,\n'
          '"cashier_at_desk_end": <true/false>,\n'
          '"cashier_leaves_desk_t": <printed time (seconds) of the first frame where the cashier has stood up and left the desk, or null if the cashier stays at the desk or is absent in all frames>,\n'
          '"client_at_desk_any": <true/false>,\n'
          '"client_at_desk_end": <true/false>,\n'
          '"cashier_actions": <list of actions seen across the frames, from: idle, counting_machine, counting_hand, handing_cash, receiving_cash, checking_document, at_safe, writing_or_typing, away, other>,\n'
          '"money_on_desk_any": <true/false: banknotes on the desk surface>,\n'
          '"money_on_desk_end": <true/false>,\n'
          '"bag_on_desk_end": <true/false: cash collection bag on the desk in the last frame>,\n'
          '"collector_bag_any": <true/false: a cash collection bag (fabric sack or pouch, often green/blue, may be sealed) is visible in ANY frame, in someone\'s hands or on the desk>,\n'
          '"personal_items_on_desk_any": <true/false: handbag, backpack, jacket, coat on the desk>,\n'
          '"food_drink_on_desk_any": <true/false: cup, bottle, snack, food on the desk>,\n'
          '"cashier_phone_in_hand_any": <true/false: the CASHIER (bank employee) holds or uses a smartphone; phones held by visitors do not count>,\n'
          '"document_shown_any": <true/false: an ID card/passport/paper is handed over or examined>,\n'
          '"safe_door_open_any": <true/false>,\n'
          '"safe_door_open_end": <true/false>,\n'
          '"key_in_safe_lock_end": <true/false: in the last frame a key (small object, often with a light key tag) is left hanging in the lock of the safe door; false if unsure>,\n'
          '"laptop_screen_end": <one of: work (bright screen with windows/applications/text), lock (login or lock screen: mostly uniform wallpaper, clock, avatar or password field), off (dark screen), closed (lid closed), unknown>,\n'
          '"door_open_end": <true/false: in the last frame the entrance door leaf is open or ajar, i.e. swung and not flush with the glass wall>,\n'
          '"lights_on_end": <true/false>,\n'
          '"deviation": <one short sentence describing any action with cash, valuables, the safe or room access that looks like a departure from a standard bank cashier procedure and is NOT covered by the fields above; otherwise "none">\n'
          "}"
    )


def _b(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "да", "1", "open", "on")
    return None


def _i(v):
    try:
        return int(round(float(v)))
    except Exception:
        return None


def _enum(v, allowed):
    if v is None:
        return None
    s = str(v).strip().lower().replace(" ", "_")
    for a in allowed:
        if s == a or s.startswith(a):
            return a
    return "other" if "other" in allowed else ("unknown" if "unknown" in allowed else None)


def normalize_facts(obj: dict) -> dict:
    """Приводит ответ одиночного кадра к типам."""
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        kk = k.strip().lower()
        if kk == "people_count":
            out[kk] = _i(v)
        elif kk == "cashier_action":
            out[kk] = _enum(v, ACTIONS)
        elif kk == "laptop_screen":
            out[kk] = _enum(v, SCREEN)
        else:
            out[kk] = _b(v)
    return out


def normalize_segment(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        kk = k.strip().lower()
        if kk.startswith("people_count"):
            out[kk] = _i(v)
        elif kk == "cashier_leaves_desk_t":
            try:
                out[kk] = None if v in (None, "null", "", "none") else float(str(v).lower().replace("s", "").replace("t=", ""))
            except Exception:
                out[kk] = None
        elif kk == "cashier_actions":
            if isinstance(v, str):
                v = [x for x in v.replace(";", ",").split(",")]
            out[kk] = [a for a in (_enum(x, ACTIONS) for x in (v or [])) if a]
        elif kk == "laptop_screen_end":
            out[kk] = _enum(v, SCREEN)
        elif kk == "deviation":
            s = str(v).strip() if v is not None else "none"
            out[kk] = "none" if s.lower() in ("none", "", "null", "no", "нет") else s
        else:
            out[kk] = _b(v)
    return out


@dataclass
class Segment:
    t0: float
    t1: float
    frames: list[Frame]
    facts: dict = field(default_factory=dict)
    raw: str = ""

    @property
    def ts(self):
        return [f.t for f in self.frames]

    def to_dict(self):
        return {"t0": self.t0, "t1": self.t1, "ts": self.ts, "facts": self.facts}


def build_segments(vf: VideoFrames, timeline: PersonTimeline | None, seg_len: float = 8.0,
                   frames_per_seg: int = 4, min_len: float = 3.0, max_segs: int = 16) -> list[Segment]:
    """Режем видео на сегменты: границы — изменения числа людей (устойчивые) + шаг seg_len."""
    bounds = {0.0, vf.duration}
    if timeline:
        for a, b, c in timeline.segments():
            if b - a >= 1.5:
                bounds.add(a)
    bounds = sorted(bounds)
    # добавляем регулярный шаг
    pts = []
    for a, b in zip(bounds, bounds[1:]):
        t = a
        while t < b:
            pts.append(t)
            t += seg_len
    pts.append(vf.duration)
    # сливаем слишком короткие
    merged = [pts[0]]
    for p in pts[1:]:
        if p - merged[-1] < min_len and p != vf.duration:
            continue
        merged.append(p)
    if len(merged) >= 2 and merged[-1] - merged[-2] < min_len and len(merged) > 2:
        merged.pop(-2)
    # бюджет вызовов: сливаем самые короткие соседние сегменты, пока их не станет <= max_segs
    while len(merged) - 1 > max_segs and len(merged) > 2:
        lens = [merged[i + 1] - merged[i] for i in range(len(merged) - 1)]
        i = min(range(len(lens)), key=lambda k: lens[k])
        # убираем границу между сегментом i и его более коротким соседом
        if i == 0:
            merged.pop(1)
        elif i == len(lens) - 1:
            merged.pop(i)
        else:
            merged.pop(i if lens[i - 1] <= lens[i + 1] else i + 1)
    segs = []
    for a, b in zip(merged, merged[1:]):
        fr = vf.between(a, b) or [vf.at(a)]
        if len(fr) > frames_per_seg:
            idx = [round(i * (len(fr) - 1) / (frames_per_seg - 1)) for i in range(frames_per_seg)]
            fr = [fr[i] for i in idx]
        segs.append(Segment(t0=a, t1=b, frames=fr))
    return segs


def label_frame(img: Image.Image, text: str, max_w: int | None = None) -> Image.Image:
    im = img.copy()
    if max_w and im.width > max_w:
        k = max_w / im.width
        im = im.resize((max_w, round(im.height * k)), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 10 + 7 * len(text), 16), fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 0))
    return im


def segment_images(seg: Segment, scene: Scene | None, frame_w: int | None = 416,
                   crops: tuple[str, ...] = ("safe", "laptop", "desk")) -> tuple[list[Image.Image], bool]:
    imgs = [label_frame(f.image, f"t={f.t:.1f}s", frame_w) for f in seg.frames]
    has_crops = False
    if scene:
        last = seg.frames[-1].image
        for name in crops:
            if scene.has(name):
                c = crop_box(last, scene.rois[name], pad=0.2, upscale=1.5, min_size=200)
                if c.width > 480:
                    c = c.resize((480, round(c.height * 480 / c.width)), Image.BILINEAR)
                imgs.append(label_frame(c, name))
                has_crops = True
    return imgs, has_crops


def video_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(0, 2)
        h.update(str(f.tell()).encode())
    return h.hexdigest()[:16]


def extract_facts(vlm: VLMBackend, vf: VideoFrames, scene: Scene | None, timeline: PersonTimeline | None,
                  seg_len: float = 8.0, frames_per_seg: int = 4, frame_w: int | None = 416,
                  max_segs: int = 16, cache_key: str | None = None, progress=None, adaptive: int = 48) -> list[Segment]:
    cache_file = None
    # сегментация и адаптивная плотность кадров определяются до обращения к кешу (имя кеша зависит от плотности)
    segs = build_segments(vf, timeline, seg_len, frames_per_seg, max_segs=max_segs)
    fps_seg = frames_per_seg
    if adaptive:
        fps_seg = max(frames_per_seg, min(8, int(adaptive) // max(1, len(segs))))
        if fps_seg != frames_per_seg:
            segs = build_segments(vf, timeline, seg_len, fps_seg, max_segs=max_segs)
    if cache_key:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{cache_key}__{vlm.name}_{getattr(vlm, 'model_id', 'm').split('/')[-1]}_{PROMPT_VERSION}_s{seg_len}_f{fps_seg}_w{frame_w}.json"
        if cache_file.exists():
            data = json.load(open(cache_file))
            segs = []
            for d in data:
                fr = [vf.at(t) for t in d["ts"]]
                segs.append(Segment(d["t0"], d["t1"], fr, d["facts"], d.get("raw", "")))
            log.info("facts from cache %s", cache_file.name)
            return segs
    for i, seg in enumerate(segs):
        imgs, has_crops = segment_images(seg, scene, frame_w)
        prompt = segment_prompt(seg.ts, has_crops)
        obj = vlm.ask_json(imgs, prompt, max_tokens=420, retries=1)
        seg.facts = normalize_segment(obj) if obj else {}
        seg.raw = vlm.stats.log[-1].get("text", "") if vlm.stats.log else ""
        log.info("seg %d/%d [%.1f-%.1f] %s", i + 1, len(segs), seg.t0, seg.t1, json.dumps(seg.facts, ensure_ascii=False)[:300])
        if progress:
            progress(i + 1, len(segs))
    if cache_file:
        json.dump([{**s.to_dict(), "raw": s.raw} for s in segs], open(cache_file, "w"), ensure_ascii=False, indent=1)
    return segs


CONFIRM_PROMPT = (
    SCENE_INTRO
    + "\nA previous pass over other frames of the same video made this claim:\n\"{claim}\"\n"
      "Look at these frames (time printed on each) and decide whether the claim is CLEARLY visible here. "
      "Be strict: if the frames do not clearly show it, answer false.\n"
      'Answer with JSON only: {{"confirmed": <true/false>, "what_is_visible": "<one short sentence>"}}'
)


def confirm_deviations(vlm: VLMBackend, vf: VideoFrames, segs: list[Segment], frame_w: int | None = 416,
                       max_checks: int = 3) -> list[dict]:
    """Второй проход по сегментам с 'deviation': другое подмножество кадров, строгий вопрос да/нет."""
    out = []
    cands = [s for s in segs if s.facts.get("deviation") and s.facts["deviation"] != "none"]
    for seg in cands[:max_checks]:
        # кадры со сдвигом на 0.5 с относительно исходных
        alt = [vf.at(t + 0.5) for t in seg.ts]
        imgs = [label_frame(f.image, f"t={f.t:.1f}s", frame_w) for f in alt]
        obj = vlm.ask_json(imgs, CONFIRM_PROMPT.format(claim=seg.facts["deviation"]), max_tokens=120)
        ok = bool(obj and _b(obj.get("confirmed")))
        out.append({"t0": seg.t0, "t1": seg.t1, "claim": seg.facts["deviation"], "confirmed": ok,
                    "visible": (obj or {}).get("what_is_visible")})
        log.info("deviation check [%.1f-%.1f] %s -> %s", seg.t0, seg.t1, seg.facts["deviation"], ok)
    return out


ROI_PROMPTS = {
    "safe": ("This is a zoomed crop of the metal safe in a bank cash office (the full frame is given for context). "
             'Answer JSON only: {"safe_door_open": <true/false: the safe door is open or ajar>, '
             '"key_in_safe_lock": <true/false: a key (small object, often with a light key tag) is left hanging in the lock of the safe door; false if unsure>, '
             '"cashier_handling_safe": <true/false: in the full frame the cashier is turned towards the safe and is handling it right now (opening, closing, putting in or taking out items); false if the cashier sits facing the desk/laptop with the safe open behind or beside them>}'),
    "laptop": ("This is a zoomed crop of the laptop in a bank cash office (the full frame is given for context). "
               'Answer JSON only: {"laptop_screen": <one of: work (bright screen showing windows/apps/text), lock (login or lock screen: mostly uniform wallpaper, clock, avatar or password field), off (dark screen), closed (lid closed), unknown>}'),
    "desk": ("This is a zoomed crop of the cashier's desk in a bank cash office (the full frame is given for context). "
             'Answer JSON only: {"money_on_desk": <true/false: banknotes lying on the desk>, '
             '"bag_on_desk": <true/false: a cash collection bag or pouch on the desk>, '
             '"documents_on_desk": <true/false: papers, ID card or documents on the desk>, '
             '"personal_items_on_desk": <true/false: handbag, backpack, jacket or coat on the desk>, '
             '"food_drink_on_desk": <true/false: cup, bottle, snack or food on the desk>}'),
}


def roi_check(vlm: VLMBackend, frame: Frame, scene: Scene | None, names=("safe", "laptop", "desk"),
              frame_w: int | None = 416) -> dict:
    """Точечные проверки по кропам ROI (по одному вызову на ROI) — точнее сегментных фактов для мелких деталей."""
    out = {"t": frame.t}
    full = label_frame(frame.image, f"t={frame.t:.1f}s", frame_w)
    for name in names:
        if not scene or not scene.has(name):
            continue
        c = crop_box(frame.image, scene.rois[name], pad=0.25, upscale=2.0, min_size=256)
        if c.width > 520:
            c = c.resize((520, round(c.height * 520 / c.width)), Image.BILINEAR)
        obj = vlm.ask_json([label_frame(c, name), full], ROI_PROMPTS[name], max_tokens=100)
        if isinstance(obj, list):
            obj = next((o for o in obj if isinstance(o, dict)), {})
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            kk = k.strip().lower()
            out[kk] = _enum(v, SCREEN) if kk == "laptop_screen" else _b(v)
    return out


def leave_check(vlm: VLMBackend, frame: Frame, scene: Scene | None, frame_w: int | None = 416) -> dict:
    return roi_check(vlm, frame, scene, ("safe", "laptop", "desk"), frame_w)


ID_PROMPT = (
    SCENE_INTRO
    + "\nThese are consecutive frames (1 per second, time printed) from the START of a client visit at the cashier's desk. "
      "Distinguish an ID document (small plastic card or passport booklet, handed to the cashier who looks at it) from banknotes (paper money, usually a stack). "
      "Answer with JSON only:\n{\n"
      '"id_document_shown": <true/false: the client hands an ID card/passport to the cashier or the cashier examines one>,\n'
      '"id_time": <printed time of the first frame where the ID document is handed/examined, or null>,\n'
      '"cash_handed": <true/false: the client hands banknotes to the cashier or puts them on the desk>,\n'
      '"cash_time": <printed time of the first frame where cash is handed, or null>\n}'
)

COUNT_PROMPT = (
    SCENE_INTRO
    + "\nThese are consecutive frames (time printed) around a moment when the cashier handles banknotes. "
      "The banknote counting machine is the black device on the desk; using it means putting a stack into it (the machine then counts). "
      "Answer with JSON only:\n{\n"
      '"machine_used": <true/false: the cashier puts banknotes into the counting machine in any frame>,\n'
      '"hand_counting": <true/false: the cashier counts banknotes by hand (fanning/flipping notes one by one) in any frame>,\n'
      '"hand_counting_time": <printed time of the first frame with hand counting, or null>,\n'
      '"what_is_visible": "<one short sentence>"\n}'
)


def _t(v):
    try:
        return None if v in (None, "null", "", "none") else float(str(v).lower().replace("s", "").replace("t=", ""))
    except Exception:
        return None


def window_check(vlm: VLMBackend, vf: VideoFrames, t0: float, t1: float, prompt: str, fps: float = 1.0,
                 frame_w: int | None = 416, max_frames: int = 8) -> dict:
    """Точечная проверка окна [t0, t1]: кадры с шагом 1/fps с подписями времени + заданный промпт."""
    ts = []
    t = max(0.0, t0)
    while t <= min(t1, vf.duration - 0.05) and len(ts) < max_frames:
        ts.append(t)
        t += 1.0 / fps
    frames = [vf.at(x) for x in ts]
    imgs = [label_frame(f.image, f"t={f.t:.1f}s", frame_w) for f in frames]
    obj = vlm.ask_json(imgs, prompt, max_tokens=160)
    if isinstance(obj, list):
        obj = next((o for o in obj if isinstance(o, dict)), {})
    if not isinstance(obj, dict):
        return {}
    out = {"t0": t0, "t1": t1}
    for k, v in obj.items():
        kk = k.strip().lower()
        if kk.endswith("_time"):
            out[kk] = _t(v)
        elif kk in ("what_is_visible", "cashier_action_after"):
            out[kk] = str(v)
        else:
            out[kk] = _b(v)
    return out


CHANGE_PROMPT = (
    SCENE_INTRO
    + "\nThese are consecutive frames (1 per second, time printed) around a moment when the person at the client side of the desk may change. "
      "Answer with JSON only:\n{\n"
      '"visitor_changed": <true/false: the first visitor leaves the room and a DIFFERENT person (different clothes/appearance) comes to the desk within these frames; false if it is the same person who stepped aside or the detector simply lost them>,\n'
      '"money_on_desk_when_cashier_alone": <true/false: banknotes lie on the desk in frames where the cashier is alone between the two visitors>,\n'
      '"cashier_put_money_in_safe": <true/false: the cashier opens the safe or puts banknotes away between the two visitors>,\n'
      '"what_is_visible": "<one short sentence>"\n}'
)


LEFT_PROMPT = (
    SCENE_INTRO
    + "\nThese are consecutive frames (1 per second, time printed) around the moment the visitor leaves the desk. "
      "Answer with JSON only:\n{\n"
      '"client_left": <true/false: the visitor stands up / walks away from the desk towards the door in these frames>,\n'
      '"cashier_counting_after": <true/false: AFTER the visitor has walked away the cashier is actively counting the banknotes: feeding a stack into the counting machine or flipping/fanning notes one by one. False if the cashier only types, writes, puts money into the safe/drawer, holds a stack without counting, or sits idle>,\n'
      '"cashier_action_after": <one of: counting_machine, counting_hand, typing_or_writing, putting_money_away, idle, other — what the cashier does after the visitor walked away>,\n'
      '"client_left_time": <printed time of the first frame where the visitor is walking away, or null>,\n'
      '"what_is_visible": "<one short sentence>"\n}'
)

ROI_PROMPTS["door"] = (
    "This is a zoomed crop of the entrance door of the room (glass door in the glass wall; the full frame is given for context). "
    'Answer JSON only: {"door_open": <true/false: the door leaf is open or ajar, i.e. swung and not flush with the glass wall>, '
    '"person_in_doorway": <true/false: a person is standing in the doorway>}'
)
