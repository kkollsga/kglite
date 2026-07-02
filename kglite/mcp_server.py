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
*factory*: the server calls it only when a manifest declares a Python embedder
library (`extensions.embedder.library: fastembed` / `sentence-transformers` /
a `factory:` escape), handing it the config as JSON; the factory lazily builds
the model (see :mod:`kglite._mcp_embed`). The user installs whichever library
they name (`pip install sentence-transformers`, etc.).
"""

from __future__ import annotations

import json
import os
import sys


def _embedder_factory(config_json: str):
    """Build a Python embedder from the `extensions.embedder` config (JSON).

    Imported lazily — only the Rust server, on seeing a Python embedder library
    in the manifest, calls this; a plain server with no embedder never imports
    any embedding library.
    """
    from kglite._mcp_embed import build_embedder

    return build_embedder(config_json)


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
    # Tell the Rust server how to re-spawn *itself* for `--selftest`. On the
    # cargo standalone binary `current_exe()` is the server, so the server
    # re-spawns directly. In the wheel the running process is the Python
    # interpreter and `kglite-mcp-server` is a console-script shim, so
    # `current_exe()` is Python — re-spawning it with server flags would launch
    # `python --graph …` and fail. Hand the server the correct vector: this
    # same module entry, run under the exact interpreter we're in. The child
    # inherits (and harmlessly re-sets) this; only the parent's `--selftest`
    # branch consumes it.
    os.environ["KGLITE_MCP_RESPAWN"] = json.dumps([sys.executable, "-m", "kglite.mcp_server"])
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
