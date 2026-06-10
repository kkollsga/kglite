"""extensions.embedder builder — fastembed Python backend.

Mirrors `crates/kglite-mcp-server/src/main.rs::build_embedder_from_manifest`
and the kglite Rust trait `Embedder { dimension, embed, load, unload }`.
The kglite Python API's `g.set_embedder(model)` requires a duck-typed
object with `dimension: int` and `embed(texts) -> list[list[float]]`
(see kglite/__init__.pyi).

We wrap `fastembed.TextEmbedding` with lazy load + unload semantics so
the ~1 GB of ONNX weights can be released between queries. Same model
catalogue as the Rust fastembed adapter (BAAI/bge-m3 et al), so the
operator's YAMLs don't change.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("kglite.mcp_server.embedder")

# Map of accepted YAML model names → dimensions. Mirrors
# `src/graph/embedder/fastembed.rs::resolve_model` so operators can
# move between Rust and Python implementations without YAML edits.
KNOWN_MODELS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
    "multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-base": 768,
    "multilingual-e5-base": 768,
}


class FastEmbedAdapter:
    """Duck-typed Embedder for kglite. Lazy-init on first `embed()`."""

    def __init__(self, model_name: str) -> None:
        if model_name not in KNOWN_MODELS:
            raise ValueError(f"unsupported fastembed model name: {model_name!r}. Known: {sorted(KNOWN_MODELS.keys())}")
        self._model_name = model_name
        self.dimension = KNOWN_MODELS[model_name]
        self._inner: Any = None

    def load(self) -> None:
        if self._inner is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as e:
                raise SystemExit(
                    "The manifest declares an `extensions.embedder`, which "
                    "requires the [embed] extra. Install with:\n"
                    "  pip install 'kglite[embed]'"
                ) from e

            self._inner = TextEmbedding(model_name=self._model_name)

    def unload(self) -> None:
        # Drop the ONNX session to free GPU/CPU memory between idle
        # periods. Re-materialises from `~/.cache/fastembed/` on next
        # load — no re-download.
        self._inner = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.load()
        # fastembed returns an iterator of numpy arrays.
        return [vec.tolist() for vec in self._inner.embed(texts)]


def from_manifest_value(value: Any) -> Any:
    """Parse `extensions.embedder` from the manifest. Returns None if
    the block is absent. bge-m3 routes through BgeM3Embedder (direct
    onnxruntime + hf-hub) because fastembed-python's catalog doesn't
    carry it. Other supported models route through fastembed-python."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"extensions.embedder must be a mapping (got {value!r})")
    backend = value.get("backend", "fastembed")
    if backend != "fastembed":
        raise ValueError(f"extensions.embedder.backend = {backend!r} is not supported. Known: fastembed.")
    model = value.get("model")
    if not model:
        raise ValueError("extensions.embedder.model is required for the fastembed backend")
    # `cooldown`: seconds to keep the model resident after the last
    # embed() call before releasing the session. Default 15 min (the
    # bge-m3 default). 0 disables cool-down (model stays resident).
    cooldown_raw = value.get("cooldown")
    cooldown: int | None = None
    if cooldown_raw is not None:
        if not isinstance(cooldown_raw, int) or cooldown_raw < 0:
            raise ValueError(f"extensions.embedder.cooldown must be a non-negative integer (got {cooldown_raw!r})")
        cooldown = cooldown_raw
    if model == "BAAI/bge-m3":
        from kglite.mcp_server.bge_m3 import BgeM3Embedder

        adapter: Any = BgeM3Embedder(cooldown_seconds=cooldown)
        log.info(
            "registered bge-m3 embedder via direct ONNX (model=%s, dim=1024, cooldown=%ss)",
            model,
            cooldown if cooldown is not None else "default-900",
        )
        return adapter
    adapter = FastEmbedAdapter(model)
    log.info("registered fastembed embedder (model=%s)", model)
    return adapter
