"""VLM через mlx-vlm (Apple Silicon). Модель задаётся HF-идентификатором (mlx-community/...-4bit)."""
from __future__ import annotations

import logging
import threading

from PIL import Image

from .base import VLMBackend, strip_thinking

log = logging.getLogger(__name__)


class MLXBackend(VLMBackend):
    name = "mlx"

    def __init__(self, model_id: str, max_pixels: int | None = None):
        super().__init__()
        import mlx.core as mx  # noqa: F401
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_id = model_id
        log.info("loading %s", model_id)
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)
        self.lock = threading.Lock()
        self.model_type = getattr(self.config, "model_type", None) or (
            self.config.get("model_type") if isinstance(self.config, dict) else None)

    def _chat_prompt(self, text: str, n_images: int) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template
        kwargs = {}
        if self.model_type and "qwen3_5" in str(self.model_type):
            kwargs["enable_thinking"] = False
        try:
            return apply_chat_template(self.processor, self.config, text, num_images=n_images, **kwargs)
        except TypeError:
            return apply_chat_template(self.processor, self.config, text, num_images=n_images)

    def _generate(self, images: list[Image.Image], prompt: str, max_tokens: int):
        from mlx_vlm import generate
        imgs = [im.convert("RGB") for im in images]
        chat = self._chat_prompt(prompt, len(imgs))
        with self.lock:
            out = generate(self.model, self.processor, chat, imgs if imgs else None,
                           max_tokens=max_tokens, temperature=0.0, verbose=False)
        if hasattr(out, "text"):
            text, pt, gt = out.text, getattr(out, "prompt_tokens", 0), getattr(out, "generation_tokens", 0)
        else:
            text, pt, gt = str(out), 0, 0
        return strip_thinking(text), int(pt or 0), int(gt or 0)
