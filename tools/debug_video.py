"""Отладка одного видео: ROI сцены (картинка), шкала людей, сегменты и факты, найденные нарушения.
Запуск: python tools/debug_video.py X-22 [--no-cache]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent


def main():
    vid = sys.argv[1]
    no_cache = "--no-cache" in sys.argv
    from pipeline.analyze import Analyzer
    from pipeline.scene import Scene, draw_scene
    from pipeline.video import sample_frames
    an = Analyzer()
    path = str(ROOT / "data/open/videos" / f"{vid}.mov")
    res = an.analyze(path, params={"use_cache": not no_cache})
    out = ROOT / "data/cache/debug"
    out.mkdir(parents=True, exist_ok=True)
    m = res["meta"]
    if m.get("scene"):
        vf = sample_frames(path, fps=1.0)
        img = draw_scene(vf.at(1.0).image, Scene.from_dict(m["scene"]))
        img.save(out / f"{vid}_scene.jpg", quality=85)
        print("scene ->", out / f"{vid}_scene.jpg", {k: [round(x, 2) for x in v] for k, v in m["scene"]["rois"].items()})
    tl = m.get("people_timeline")
    if tl:
        print("people:", "".join(str(c) for c in tl["count"]))
    for s in m["segments"]:
        print(f"[{s['t0']:5.1f}-{s['t1']:5.1f}] {json.dumps(s['facts'], ensure_ascii=False)}")
    print("deviations:", m.get("deviations"))
    print("leave_checks:", m.get("leave_checks"))
    print("timings:", m["timings"])
    print("VERDICT:", res["verdict"])
    for v in res["violations"]:
        print(f"  {v['type_id']} @ {v['timecode_sec']} conf={v['confidence']} :: {v['description'][:120]}")
    slim = {**res, "violations": [{k: x for k, x in v.items() if k != "evidence_frames"} for v in res["violations"]]}
    json.dump(slim, open(out / f"{vid}_result.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
