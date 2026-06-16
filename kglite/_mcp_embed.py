"""fastembed-py embedder for the bundled MCP server's ``backend: python``.

The ``kglite-mcp-server`` console-script shim (``kglite/mcp_server.py``) hands
:func:`build_embedder` to ``kglite._run_mcp_server`` as the embedder factory.
The Rust server invokes it **only** when a manifest declares
``extensions.embedder.backend: python`` — so ``fastembed`` is imported lazily,
never at plain server boot. The returned object satisfies kglite's
``EmbeddingModel`` protocol (``dimension`` + ``embed``); the server wraps it in
a ``PyEmbedderAdapter`` and calls it once per ``text_score()`` query.

This is what lets ``pip install 'kglite[embed]'`` power semantic search in the
bundled server with no Rust toolchain and no ``ort-sys`` download. (The
standalone ``cargo install`` binary uses ``backend: fastembed`` / fastembed-rs
instead — it has no Python.)
"""

from __future__ import annotations


class _FastEmbedModel:
    """Adapts fastembed's ``TextEmbedding`` to kglite's ``EmbeddingModel``
    protocol (``dimension: int`` + ``embed(texts) -> list[list[float]]``)."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without [embed]
            raise RuntimeError(
                "extensions.embedder.backend: python needs the fastembed package. "
                "Install it with: pip install 'kglite[embed]'."
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        # The protocol requires `dimension` up front, but fastembed only
        # reveals it per query — probe once with a trivial input.
        probe = next(iter(self._model.embed(["x"])))
        self.dimension = int(len(probe))

    def embed(self, texts: list[str]) -> list[list[float]]:
        # fastembed yields one ndarray per text; coerce to plain float lists
        # so the PyEmbedderAdapter's `list[list[float]]` extract succeeds.
        return [[float(x) for x in vec] for vec in self._model.embed(list(texts))]


def build_embedder(model_name: str) -> _FastEmbedModel:
    """Factory handed to ``kglite._run_mcp_server``. Builds a fastembed-py model
    for ``model_name``; any error surfaces to the server as a boot failure with
    a clear message."""
    return _FastEmbedModel(model_name)
