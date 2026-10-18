"""Мини-бенчмарк VLM-кандидатов: точность по наблюдаемым фактам на размеченных кадрах + скорость.
Запуск: python tools/bench_vlm.py mlx-community/Qwen3.5-9B-4bit [--sheet]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image  # noqa: E402

from pipeline.vlm.base import make_backend  # noqa: E402
from pipeline.facts import FACT_PROMPT_SINGLE, normalize_facts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "cache" / "bench"


def match(fact, exp, got):
    if got is None:
        return False
    if fact == "people_count":
        try:
            return int(got) == int(exp)
        except Exception:
            return False
    if isinstance(exp, bool):
        return bool(got) == exp
    return str(got).lower() == str(exp).lower()


def main():
    model_id = sys.argv[1]
    vlm = make_backend(model_id)
    items = json.load(open(BENCH / "items.json"))
    rows = []
    ok = 0
    # прогрев
    vlm.ask([Image.open(items[0]["path"])], "Describe the image in 5 words.", max_tokens=20)
    t_all = time.time()
    for it in items:
        img = Image.open(it["path"]).convert("RGB")
        t0 = time.time()
        obj = vlm.ask_json([img], FACT_PROMPT_SINGLE, max_tokens=350)
        dt = time.time() - t0
        facts = normalize_facts(obj) if obj else {}
        got = facts.get(it["fact"])
        hit = match(it["fact"], it["expected"], got)
        ok += hit
        rows.append({**it, "got": got, "hit": hit, "dt": round(dt, 1)})
        print(f"{'OK ' if hit else 'ERR'} {it['video']}@{it['t']}s {it['fact']}: exp={it['expected']} got={got} ({dt:.1f}s)")
    total = time.time() - t_all
    st = vlm.stats
    summary = {"model": model_id, "acc": ok / len(items), "n": len(items), "sec_per_call": total / len(items),
               "prompt_tok_per_call": st.prompt_tokens / max(1, st.calls),
               "gen_tok_per_call": st.gen_tokens / max(1, st.calls)}
    print(json.dumps(summary, ensure_ascii=False))
    out = ROOT / "reports" / f"bench_{model_id.split('/')[-1]}.json"
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
