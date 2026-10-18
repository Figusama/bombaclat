"""Готовит кадры для мини-бенчмарка VLM: (видео, секунда, вопрос, ожидаемый ответ)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.video import sample_frames

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cache" / "bench"
OUT.mkdir(parents=True, exist_ok=True)

# (video, t, fact, expected)  — ожидания из таблицы DATATHON (1).csv и разметки
ITEMS = [
    ("X-22", 16, "key_in_safe_lock", True),   ("X-23", 19, "key_in_safe_lock", False),
    ("X-15", 17, "key_in_safe_lock", False),
    ("X-28", 9, "laptop_screen", "work"),      ("X-29", 12, "laptop_screen", "lock"),
    ("X-22", 17, "laptop_screen", "lock"),
    ("X-20", 7, "safe_door_open", True),       ("X-21", 9, "safe_door_open", False),
    ("X-18", 21, "people_count", 3),           ("X-30", 14, "people_count", 2),
    ("X-5", 12, "people_count", 2),            ("X-11", 3, "people_count", 1),
    ("X-28", 10, "people_count", 0),
    ("X-26", 10, "money_on_desk", True),       ("X-27", 12, "money_on_desk", False),
    ("X-45", 10, "money_on_desk", True),
    ("X-32", 7, "door_open", True),            ("X-31", 8, "door_open", False),
    ("X-11", 19, "lights_on", False),          ("X-11", 3, "lights_on", True),
    ("X-40", 17, "personal_items_on_desk", True), ("X-25", 6, "personal_items_on_desk", False),
    ("X-46", 11, "food_drink_on_desk", True),  ("X-25", 6, "food_drink_on_desk", False),
    ("X-4", 62, "cashier_action", "counting_hand"), ("X-44", 16, "cashier_action", "counting_hand"),
    ("X-35", 12, "cashier_action", "counting_machine"),
    ("X-15", 15, "bag_on_desk", True),         ("X-23", 19, "bag_on_desk", False),
]

cache = {}
meta = []
for i, (vid, t, fact, exp) in enumerate(ITEMS):
    if vid not in cache:
        cache[vid] = sample_frames(str(ROOT / "data/open/videos" / f"{vid}.mov"), fps=2.0)
    fr = cache[vid].at(t)
    p = OUT / f"{i:02d}_{vid}_{t}s.jpg"
    fr.image.save(p, quality=90)
    meta.append({"i": i, "video": vid, "t": fr.t, "fact": fact, "expected": exp, "path": str(p)})
json.dump(meta, open(OUT / "items.json", "w"), ensure_ascii=False, indent=1)
print(len(meta), "frames ->", OUT)
