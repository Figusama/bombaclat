"""Стенд оценки кандидатов по критериям жюри.

Запуск:
  python tools/evaluate.py --candidate rules
  python tools/evaluate.py --candidate vlm_video --opt model=Qwen3-VL-4B-Instruct --opt fps=1 --split dev
Кандидат = модуль analyzer/candidates/<имя>.py с функцией analyze(video_path, **opts) -> dict по схеме ТЗ.
"""
import argparse, importlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GT = json.load(open(ROOT / "data/labels/gt.json"))
VIDEOS = ROOT / "data/open/videos"
CACHE = ROOT / "data/cache/eval"


def run_candidate(name, opts, videos, force=False):
    tag = name + ("__" + "_".join(f"{k}-{v}" for k, v in sorted(opts.items())) if opts else "")
    cdir = CACHE / tag
    cdir.mkdir(parents=True, exist_ok=True)
    mod = None
    out = {}
    for v in videos:
        f = cdir / f"{v}.json"
        if f.exists() and not force:
            out[v] = json.load(open(f))
            continue
        if mod is None:
            mod = importlib.import_module(f"analyzer.candidates.{name}")
        t = time.time()
        try:
            res = mod.analyze(str(VIDEOS / f"{v}.mov"), **opts)
        except Exception as e:  # кандидат не должен ронять стенд
            res = {"verdict": "error", "violations": [], "error": repr(e)}
        res["latency_sec"] = round(time.time() - t, 1)
        for x in res.get("violations", []):
            x.pop("evidence_frames", None)  # кадры в кэш не пишем
        json.dump(res, open(f, "w"), ensure_ascii=False, indent=1)
        out[v] = res
        print(f"  {v}: {res['verdict']} {[(x['type'], x['timecode_sec']) for x in res.get('violations', [])]} "
              f"{res['latency_sec']}s", flush=True)
    return tag, out


def metrics(videos, preds):
    n_v = sum(GT[v]["verdict"] == "violation" for v in videos)
    n_n = len(videos) - n_v
    tp = sum(GT[v]["verdict"] == "violation" and preds[v]["verdict"] == "violation" for v in videos)
    tn = sum(GT[v]["verdict"] == "no_violation" and preds[v]["verdict"] == "no_violation" for v in videos)
    t2 = t4 = ty = n_ev = 0
    for v in videos:
        pv = preds[v].get("violations", [])
        for e in GT[v]["events"]:
            if e["uncertain"]:
                continue
            n_ev += 1
            d = [abs(p["timecode_sec"] - e["t"]) for p in pv]
            t2 += bool(d) and min(d) <= 2
            t4 += bool(d) and min(d) <= 4
            ty += any(p["type"] == e["type"] and abs(p["timecode_sec"] - e["t"]) <= 4 for p in pv)
    lat = sorted(preds[v].get("latency_sec", 0) for v in videos)
    m = {
        "videos": len(videos),
        "verdict_acc": (tp + tn) / len(videos) if videos else 0,
        "recall": tp / n_v if n_v else 0,
        "specificity": tn / n_n if n_n else 0,
        "time_hit_2s": t2 / n_ev if n_ev else 0,
        "time_hit_4s": t4 / n_ev if n_ev else 0,
        "type_acc": ty / n_ev if n_ev else 0,
        "lat_p50": lat[len(lat) // 2] if lat else 0,
        "lat_max": lat[-1] if lat else 0,
    }
    m["score_55"] = 25 * m["verdict_acc"] + 15 * m["time_hit_2s"] + 15 * m["type_acc"]
    return m


def fmt(m):
    return (f"n={m['videos']:2} | verdict {m['verdict_acc']:.2f} (recall {m['recall']:.2f}, spec {m['specificity']:.2f}) | "
            f"time≤2s {m['time_hit_2s']:.2f} ≤4s {m['time_hit_4s']:.2f} | type {m['type_acc']:.2f} | "
            f"lat p50 {m['lat_p50']}s max {m['lat_max']}s | SCORE {m['score_55']:.1f}/55")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--opt", action="append", default=[], help="key=value, передаётся в analyze()")
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    opts = {}
    for kv in a.opt:
        k, v = kv.split("=", 1)
        opts[k] = float(v) if v.replace(".", "", 1).isdigit() else v
    videos = a.videos or sorted((v for v in GT if a.split == "all" or GT[v]["split"] == a.split),
                                key=lambda s: int(s.split("-")[1]))
    tag, preds = run_candidate(a.candidate, opts, videos, a.force)

    lines = [f"# {tag}\n"]
    for split in ("dev", "holdout"):
        vs = [v for v in videos if GT[v]["split"] == split]
        if vs:
            lines.append(f"{split:8} {fmt(metrics(vs, preds))}")
    lines.append(f"{'all':8} {fmt(metrics(videos, preds))}")
    for ang in "ABC":
        vs = [v for v in videos if GT[v]["angle"] == ang]
        if vs:
            lines.append(f"угол {ang}   {fmt(metrics(vs, preds))}")
    lines.append("\n## Ошибки по видео")
    for v in videos:
        g, p = GT[v], preds[v]
        gt_ev = [(e["type"], e["t"]) for e in g["events"]]
        pr_ev = [(x["type"], x["timecode_sec"]) for x in p.get("violations", [])]
        ok = g["verdict"] == p["verdict"]
        if not ok or gt_ev:
            lines.append(f"{'OK ' if ok else 'ERR'} {v:5} [{g['split']}/{g['angle']}] GT {gt_ev or 'нет'} | PRED {pr_ev or p['verdict']}"
                         + (f" | {p['error']}" if p.get("error") else ""))
    report = "\n".join(lines)
    print("\n" + report)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"eval_{tag}.md").write_text(report)


if __name__ == "__main__":
    main()
