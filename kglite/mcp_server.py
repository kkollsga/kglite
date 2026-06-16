"""Console-script shim for the bundled ``kglite-mcp-server``.

The MCP server itself is pure Rust — it lives in the ``kglite-mcp-server``
*library* and is statically linked into this wheel's compiled extension,
sharing the one ``kglite`` engine (no separate wheel, no duplicated engine).
This module is the thin Python entry point wired to the ``kglite-mcp-server``
console script in ``pyproject.toml``; it does nothing but forward the
command-line arguments into the Rust ``run`` function.

Equivalent invocations::

    pip install kglite   &&  kglite-mcp-server --graph foo.kgl
    cargo install kglite-mcp-server  &&  kglite-mcp-server --graph foo.kgl

Both run the identical Rust server. Argument parsing happens Rust-side (clap),
so this shim stays a forwarder. The one thing it supplies is an embedder
*factory*: the server calls it only if a manifest declares
``extensions.embedder.backend: python``, at which point the factory lazily
builds a fastembed-py model (so ``pip install 'kglite[embed]'`` powers
``text_score()`` in the bundled server — see :mod:`kglite._mcp_embed`).
"""

from __future__ import annotations

import sys


def _embedder_factory(model_name: str):
    """Build a fastembed-py embedder for ``extensions.embedder.backend: python``.

    Imported lazily — only the Rust server, on seeing that manifest backend,
    calls this; a plain server with no embedder never touches ``fastembed``.
    """
    from kglite._mcp_embed import build_embedder

    return build_embedder(model_name)


def main(argv: list[str] | None = None) -> int:
    """Forward ``argv`` (default ``sys.argv[1:]``) to the bundled Rust server.

    Blocks until the server exits (it serves over stdio). The Rust side
    releases the GIL for the whole run, so the process simply *becomes* the
    MCP server. Returns ``0`` on a clean exit; a server error surfaces as a
    ``RuntimeError`` from the extension, which we translate to a non-zero exit
    with the message on stderr.
    """
    from kglite import _run_mcp_server

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        _run_mcp_server(args, embedder_factory=_embedder_factory)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(f"kglite-mcp-server: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
