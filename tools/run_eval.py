"""Прогон датасета и метрики. Запуск: python tools/run_eval.py [--split all|dev|holdout] [--videos X-1,X-2] [--tag name]
Метрики: verdict acc / recall / spec, таймкод в ±2/±4 с, type acc, латентность; срез по ракурсам."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
GT = json.load(open(ROOT / "tools/labels/gt.json"))
VIDEOS = ROOT / "data/open/videos"


def metrics(rows):
    n = len(rows)
    if n == 0:
        return {}
    acc = sum(r["gt_verdict"] == r["pred_verdict"] for r in rows) / n
    pos = [r for r in rows if r["gt_verdict"] == "violation"]
    neg = [r for r in rows if r["gt_verdict"] == "no_violation"]
    rec = sum(r["pred_verdict"] == "violation" for r in pos) / max(1, len(pos))
    spec = sum(r["pred_verdict"] == "no_violation" for r in neg) / max(1, len(neg))
    ev = [(r, e) for r in pos for e in r["gt_events"] if not e.get("uncertain")]
    t2 = sum(any(abs(p["t"] - e["t"]) <= 2 for p in r["pred"]) for r, e in ev) / max(1, len(ev))
    t4 = sum(any(abs(p["t"] - e["t"]) <= 4 for p in r["pred"]) for r, e in ev) / max(1, len(ev))
    ty = sum(any(p["type"] == e["type"] and abs(p["t"] - e["t"]) <= 6 for p in r["pred"]) for r, e in ev) / max(1, len(ev))
    ty_any = sum(any(p["type"] == e["type"] for p in r["pred"]) for r, e in ev) / max(1, len(ev))
    lat = sorted(r["latency"] for r in rows)
    fp_types = {}
    for r in neg:
        for p in r["pred"]:
            fp_types[p["type"]] = fp_types.get(p["type"], 0) + 1
    return {"n": n, "verdict_acc": round(acc, 2), "recall": round(rec, 2), "spec": round(spec, 2),
            "time2": round(t2, 2), "time4": round(t4, 2), "type_acc": round(ty, 2), "type_any": round(ty_any, 2),
            "lat_p50": round(lat[len(lat) // 2], 1), "lat_max": round(lat[-1], 1),
            "score55": round(25 * acc + 15 * t2 + 15 * ty, 1), "fp_types": fp_types}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    ap.add_argument("--videos", default="")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--rules-only", action="store_true", help="пересчитать только правила по кешу фактов")
    args = ap.parse_args()

    from pipeline.analyze import Analyzer
    an = Analyzer()
    vids = [v for v in args.videos.split(",") if v] or [k for k, v in GT.items() if args.split == "all" or v["split"] == args.split]
    out_dir = ROOT / "data/cache/eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in vids:
        path = VIDEOS / f"{vid}.mov"
        if not path.exists():
            print("missing", vid)
            continue
        t0 = time.time()
        res = an.analyze(str(path), params={"use_cache": not args.no_cache})
        lat = time.time() - t0
        pred = [{"type": v["type_id"], "t": v["timecode_sec"], "conf": v["confidence"]} for v in res["violations"]]
        g = GT[vid]
        row = {"video": vid, "angle": g["angle"], "split": g["split"], "gt_verdict": g["verdict"], "gt_events": g["events"],
               "pred_verdict": res["verdict"], "pred": pred, "latency": round(lat, 1), "timings": res["meta"]["timings"]}
        rows.append(row)
        slim = {k: v for k, v in res.items() if k != "violations"}
        slim["violations"] = [{k: v for k, v in x.items() if k != "evidence_frames"} for x in res["violations"]]
        json.dump(slim, open(out_dir / f"{vid}.json", "w"), ensure_ascii=False, indent=1)
        ok = "OK " if row["gt_verdict"] == row["pred_verdict"] else "ERR"
        print(f"{ok} {vid:5s} [{g['split']}/{g['angle']}] GT {[(e['type'], e['t']) for e in g['events']] or 'нет'} | "
              f"PRED {[(p['type'], p['t']) for p in pred] or 'no_violation'} | {lat:.0f}s", flush=True)

    lines = [f"# {args.tag}", ""]
    for name, sel in [("all", rows), ("dev", [r for r in rows if r["split"] == "dev"]),
                      ("holdout", [r for r in rows if r["split"] == "holdout"])] + \
                     [(f"angle {a}", [r for r in rows if r["angle"] == a]) for a in "ABC"]:
        if sel:
            lines.append(f"{name:9s} " + json.dumps(metrics(sel), ensure_ascii=False))
    lines += ["", "## По видео"]
    for r in rows:
        ok = "OK " if r["gt_verdict"] == r["pred_verdict"] else "ERR"
        lines.append(f"{ok} {r['video']} [{r['split']}/{r['angle']}] GT {[(e['type'], e['t']) for e in r['gt_events']] or 'нет'} "
                     f"| PRED {[(p['type'], p['t']) for p in r['pred']] or 'no_violation'} | {r['latency']}s")
    rep = ROOT / "reports" / f"eval_{args.tag}.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    print("report ->", rep)


if __name__ == "__main__":
    main()
