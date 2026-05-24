# MCP server: pre-0.9.20 migrations

Historical migration notes for operators upgrading from
kglite-mcp-server releases prior to 0.9.20. New users should follow
the current setup flow in {doc}`../guides/mcp-servers` instead.

## Migration: 0.9.19 → 0.9.20

`kglite-mcp-server` is now a Python entry point instead of a bundled
Rust binary. Operator action:

```bash
pip install --upgrade 'kglite[mcp]'
```

YAMLs unchanged. Tool surface unchanged. fastembed cache directory
unchanged (`~/.cache/fastembed/`). Performance unchanged (kglite's
Python `cypher()` releases the GIL for execution, so the wrapping
layer is sub-microsecond).

What disappeared:
- `kglite/_bin/kglite-mcp-server` binary inside the wheel (no
  longer built).
- `install_name_tool` / `patchelf` / mold / per-Python-version
  wheel matrix in CI (no longer needed).
- The 0.9.18 conda install_name regression (impossible by
  construction — there's no binary to mis-link).

Wheel matrix is back to 3 abi3 wheels per release, same as pre-0.9.18.

## Migration: 0.9.17 → 0.9.18

### Embedders: `embedder:` → `extensions.embedder:`

The framework-level `embedder:` block (Python class factory) is gone.
Replace with `extensions.embedder:` (Rust-native fastembed-rs):

```yaml
# Before (0.9.17 and earlier — no longer parsed)
embedder:
  module: ./embedder.py
  class: BgeM3Embedder
trust:
  allow_embedder: true

# After (0.9.18+)
extensions:
  embedder:
    backend: fastembed
    model: BAAI/bge-m3            # or any fastembed catalog name
```

Operators with custom `embedder.py` files don't need them any more —
fastembed-rs supports BAAI/bge-m3, bge-small/base/large-en-v1.5,
all-MiniLM-L6-v2, and the multilingual-e5 family natively, downloading
ONNX weights on first use to `~/.cache/fastembed/`.

### `tools[].python:` → Rust shim or Cypher template

Python tool hooks are removed in 0.9.18. Two replacements depending
on shape:

- If the function is mostly Cypher with light parameter munging,
  promote it to a `tools[].cypher` entry with a `$param` template.
- If it has real logic (HTTP fetch, file parse), write a small
  downstream Rust binary that embeds the kglite crate directly —
  see **Building a downstream binary** in the main guide. The
  binary calls `kglite::api::CypherExecutor` / `compute_description`
  / etc. without any Python boundary.

### Wheel install

`pip install kglite` now lands `kglite-mcp-server` on `PATH` directly.
The 0.9.17-era discovery flow (`otool -L`, `PYO3_PYTHON=`,
`install_name_tool -add_rpath`) is unnecessary. If your shell still
points at an old `cargo install` binary, drop it and let `pip` win.

### CSV-over-HTTP

The new `extensions.csv_http_server` block opts into a localhost HTTP
listener that serves `FORMAT CSV` exports as URLs instead of inline
strings:

```yaml
extensions:
  csv_http_server:
    port: 8765
    dir: temp/                    # relative to the manifest
    cors_origin: "*"              # optional, defaults to "*"
```

With this set, a `cypher_query` that ends in `FORMAT CSV` writes the
result to `temp/kglite-<hash>.csv` and returns a
`http://127.0.0.1:8765/...` URL the agent can fetch when ready.
Useful for million-row exports that would otherwise blow the MCP
response budget.
