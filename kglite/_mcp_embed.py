"""Python embedder construction for the bundled MCP server.

When a manifest declares `extensions.embedder` with a Python embedding library,
the Rust server hands the *whole* config object (as JSON) to
:func:`build_embedder` here. We pick the library, build the model, and return an
object satisfying kglite's ``EmbeddingModel`` protocol (``dimension`` +
``embed(texts) -> list[list[float]]``); the server wraps it in a
``PyEmbedderAdapter`` and calls it once per ``text_score()`` query.

Adding a library is a change **here only** — the Rust server stays agnostic
(it just asks "is this a Python library? hand it to Python"). The user installs
whichever library they name in the manifest:

```yaml
extensions:
  embedder:
    library: sentence-transformers   # pip install sentence-transformers
    model: BAAI/bge-m3               # ← works (fastembed-py has no bge-m3)
# library: fastembed              → pip install fastembed
# factory: mypkg.embed:build      → any custom builder returning an EmbeddingModel
```

(`library: fastembed-rs` is the Rust engine — handled in the cargo binary, never
reaches here.)
"""

from __future__ import annotations

import importlib
import json


class _FastEmbedModel:
    """fastembed-py `TextEmbedding` → the EmbeddingModel protocol."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without fastembed
            raise RuntimeError(
                "extensions.embedder.library 'fastembed' is not installed: pip install fastembed"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        # The protocol needs `dimension` up front; fastembed reveals it per
        # query, so probe once. (Raises here if the model name is unsupported —
        # e.g. fastembed-py has no bge-m3; use library: sentence-transformers.)
        probe = next(iter(self._model.embed(["x"])))
        self.dimension = int(len(probe))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(list(texts))]


class _SentenceTransformerModel:
    """sentence-transformers `SentenceTransformer` → the EmbeddingModel protocol.
    Loads any HuggingFace embedding model, including `BAAI/bge-m3`."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "extensions.embedder.library 'sentence-transformers' is not installed: "
                "pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(list(texts)).tolist()


#: Curated Python embedding libraries. Anything outside this set goes through
#: the `factory:` escape (so we don't grow a wrapper per library).
_LIBRARIES = {
    "fastembed": _FastEmbedModel,
    "sentence-transformers": _SentenceTransformerModel,
}


def build_embedder(config_json: str):
    """Build an `EmbeddingModel` from the `extensions.embedder` config (JSON).

    Dispatch: `factory:` (a `module:attr` builder) wins; otherwise `library:`
    (default `fastembed`) selects a curated wrapper. Errors are raised with an
    actionable message and surface to the server as a boot failure.
    """
    cfg = json.loads(config_json)

    factory = cfg.get("factory")
    if factory:
        module_path, sep, attr = str(factory).partition(":")
        if not sep:
            raise RuntimeError(f"extensions.embedder.factory must be 'module:attr', got {factory!r}")
        try:
            fn = getattr(importlib.import_module(module_path), attr)
        except Exception as exc:
            raise RuntimeError(f"extensions.embedder.factory {factory!r} failed to import: {exc}") from exc
        return fn(cfg.get("model"))

    model = cfg.get("model")
    if not model:
        raise RuntimeError("extensions.embedder.model is required")
    library = cfg.get("library", "fastembed")
    cls = _LIBRARIES.get(library)
    if cls is None:
        known = ", ".join(sorted(_LIBRARIES))
        raise RuntimeError(
            f"unknown embedder library {library!r}; known Python libraries: {known} — "
            f"or use `factory: module:attr` for any other embedder. "
            f"(For the Rust engine use `library: fastembed-rs` on the cargo "
            f"`--features fastembed` binary.)"
        )
    return cls(model)
