"""Распаковка open.zip в data/open, нормализация имён (кириллическая Х -> латинская X), удаление дубликатов."""
import filecmp, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "open.zip"
out = ROOT / "data" / "open"
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(out.parent)
raw = out / "videos"
seen = []
for p in sorted(raw.glob("*.mov")):
    name = "X-" + p.name.split("-", 1)[1]
    dst = raw / name
    if p.name != name:
        p.rename(dst)
    dup = next((s for s in seen if filecmp.cmp(s, dst, shallow=False)), None)
    if dup:
        print(f"{dst.name} duplicates {dup.name} -> removed")
        dst.unlink()
    else:
        seen.append(dst)
print(len(seen), "unique videos in", raw)
