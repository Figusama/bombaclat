"""Единый интерфейс к локальной VLM: список картинок + текст -> текст."""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class VLMStats:
    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    log: list = field(default_factory=list)


class VLMBackend(ABC):
    name: str = "base"

    def __init__(self):
        self.stats = VLMStats()

    @abstractmethod
    def _generate(self, images: list[Image.Image], prompt: str, max_tokens: int) -> tuple[str, int, int]:
        """Возвращает (text, prompt_tokens, gen_tokens)."""

    def ask(self, images: list[Image.Image], prompt: str, max_tokens: int = 400) -> str:
        t0 = time.time()
        text, pt, gt = self._generate(images, prompt, max_tokens)
        dt = time.time() - t0
        self.stats.calls += 1
        self.stats.seconds += dt
        self.stats.prompt_tokens += pt
        self.stats.gen_tokens += gt
        self.stats.log.append({"dt": round(dt, 2), "pt": pt, "gt": gt, "n_img": len(images)})
        return text

    def ask_json(self, images: list[Image.Image], prompt: str, max_tokens: int = 400,
                 retries: int = 1) -> dict | list | None:
        """Запрос с парсингом JSON из ответа (с ремонтом). None, если не удалось."""
        for attempt in range(retries + 1):
            text = self.ask(images, prompt, max_tokens)
            obj = extract_json(text)
            if obj is not None:
                return obj
            prompt = prompt + "\n\nОтветь строго валидным JSON без пояснений и без markdown."
        return None


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    # незакрытый блок размышлений
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
    return text.strip()


def extract_json(text: str):
    """Достаёт первый JSON-объект/массив из текста; чинит типичные огрехи."""
    text = strip_thinking(text)
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    # ищем сбалансированный конец
    depth = 0
    end = None
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    cand = text[start:end] if end else text[start:]
    for fix in (lambda s: s,
                lambda s: re.sub(r",\s*([}\]])", r"\1", s),
                lambda s: s.replace("'", '"'),
                lambda s: re.sub(r",\s*([}\]])", r"\1", s.replace("'", '"')),
                lambda s: s + "}" * max(0, s.count("{") - s.count("}")),
                ):
        try:
            return json.loads(fix(cand))
        except Exception:
            continue
    return None


def make_backend(model_id: str | None = None) -> VLMBackend:
    model_id = model_id or os.environ.get("MODEL_ID", "mlx-community/Qwen3.5-9B-4bit")
    backend = os.environ.get("VLM_BACKEND", "mlx")
    if backend == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend(model_id)
    if backend == "hf":
        from .hf_backend import HFBackend
        return HFBackend(model_id)
    raise ValueError(f"unknown VLM_BACKEND={backend}")
