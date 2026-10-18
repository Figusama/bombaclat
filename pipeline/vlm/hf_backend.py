"""VLM через transformers (Linux/CUDA или CPU) — для переноса во внутренний контур без Apple Silicon.
MODEL_ID: HF-идентификатор или локальный путь (например Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3.5-9B)."""
from __future__ import annotations

import logging
import threading

import torch
from PIL import Image

from .base import VLMBackend, strip_thinking

log = logging.getLogger(__name__)


class HFBackend(VLMBackend):
    name = "hf"

    def __init__(self, model_id: str, device: str | None = None):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        log.info("loading %s on %s", model_id, self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.lock = threading.Lock()

    @torch.no_grad()
    def _generate(self, images: list[Image.Image], prompt: str, max_tokens: int):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        kwargs = {}
        try:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                      enable_thinking=False)
        except TypeError:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[im.convert("RGB") for im in images] or None,
                                return_tensors="pt").to(self.device)
        with self.lock:
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, **kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        txt = self.processor.batch_decode([gen], skip_special_tokens=True)[0]
        return strip_thinking(txt), int(inputs["input_ids"].shape[1]), int(gen.shape[0])
