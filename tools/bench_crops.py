"""Проверка: помогают ли кропы ROI (сейф, ноутбук) на мелких фактах. Использует grounding на том же кадре."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from pipeline.vlm.base import make_backend
from pipeline.scene import calibrate
from pipeline.video import crop_box
from pipeline.facts import label_frame, _b, _enum, SCREEN
ROOT = Path(__file__).resolve().parent.parent
items = json.load(open(ROOT/"data/cache/bench/items.json"))
items = [it for it in items if it["fact"] in (sys.argv[2].split(",") if len(sys.argv)>2 else ("key_in_safe_lock","laptop_screen","safe_door_open","door_open"))]
vlm = make_backend(sys.argv[1] if len(sys.argv)>1 else None)
P_SAFE = ("This is a zoomed crop of the metal safe in a bank cash office (plus the full frame for context). "
          'Answer JSON only: {"safe_door_open": <true/false>, "key_in_safe_lock": <true/false: a key (small metal object, possibly with a tag) is inserted in the lock of the safe door>}')
P_LAP = ("This is a zoomed crop of the laptop in a bank cash office (plus the full frame for context). "
         'Answer JSON only: {"laptop_screen": <one of: work (bright screen showing windows/apps/text), lock (login or lock screen: mostly uniform wallpaper, clock, avatar or password field), off (dark screen), closed (lid closed), unknown>}')
P_DOOR = ("Look at the entrance door of the room (glass door in the glass wall; zoomed crop plus full frame). "
          'Answer JSON only: {"door_open": <true/false: the door leaf is open or ajar (not aligned with the glass wall)>}')
ok=n=0
scenes={}
for it in items:
    img = Image.open(it["path"]).convert("RGB")
    if it["video"] not in scenes:
        scenes[it["video"]] = calibrate(vlm, img)
    sc = scenes[it["video"]]
    f = it["fact"]
    roi = {"key_in_safe_lock":"safe","safe_door_open":"safe","laptop_screen":"laptop","door_open":"door"}[f]
    prompt = {"key_in_safe_lock":P_SAFE,"safe_door_open":P_SAFE,"laptop_screen":P_LAP,"door_open":P_DOOR}[f]
    if not sc.has(roi):
        print("no roi", roi, it["video"]); continue
    crop = crop_box(img, sc.rois[roi], pad=0.25, upscale=2.0, min_size=256)
    t0=time.time()
    obj = vlm.ask_json([label_frame(crop, roi), label_frame(img, "full frame", 416)], prompt, max_tokens=80) or {}
    if isinstance(obj, list):
        obj = next((o for o in obj if isinstance(o, dict)), {})
    got = _enum(obj.get(f), SCREEN) if f=="laptop_screen" else _b(obj.get(f))
    hit = (got == it["expected"]) ; ok+=hit; n+=1
    print(("OK " if hit else "ERR"), it["video"], f, "exp", it["expected"], "got", got, f"{time.time()-t0:.1f}s", "roi", [round(x,2) for x in sc.rois[roi]])
print("acc", ok, "/", n)
