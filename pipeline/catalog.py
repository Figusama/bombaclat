"""Загрузка каталога типов нарушений и нормативных ссылок из catalog/violations.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog" / "violations.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    docs = data["documents"]
    by_id = {}
    for t in data["types"]:
        norms = []
        for n in t.get("norms", []):
            d = docs[n["doc"]]
            norms.append({
                "document": d["title"],
                "document_short": d["short"],
                "clause": n["clause"],
                "url": d["url"],
                "quote": n["quote"],
                "basis": n.get("basis", "indirect"),
                "note": n.get("note"),
            })
        by_id[t["id"]] = {**t, "norms": norms}
    return {"documents": docs, "types": by_id, "order": [t["id"] for t in data["types"]]}


def get_type(type_id: str) -> dict:
    return load_catalog()["types"][type_id]


def primary_norm(type_id: str) -> dict:
    return get_type(type_id)["norms"][0]


def rule_text(type_id: str) -> str:
    """Короткая строка 'документ, пункт' для вывода в API/UI."""
    n = primary_norm(type_id)
    return f"{n['document_short']}, {n['clause']}"
