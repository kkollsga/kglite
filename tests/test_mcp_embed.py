"""Unit tests for `kglite._mcp_embed.build_embedder` — the Python-side embedder
dispatch for the bundled MCP server's `extensions.embedder.library`.

Dispatch + error messaging only (no model downloads / network)."""

from __future__ import annotations

import json
import sys

import pytest

from kglite import _mcp_embed


def _cfg(**kw) -> str:
    return json.dumps(kw)


def test_factory_escape_is_called(tmp_path) -> None:
    # A `factory: module:attr` builder is imported and called with the model.
    mod = tmp_path / "myembed.py"
    mod.write_text(
        "class _Stub:\n"
        "    dimension = 3\n"
        "    def embed(self, texts):\n"
        "        return [[0.0, 0.0, 0.0] for _ in texts]\n"
        "def build(model):\n"
        "    s = _Stub(); s.model = model; return s\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        obj = _mcp_embed.build_embedder(_cfg(factory="myembed:build", model="anything"))
    finally:
        sys.path.remove(str(tmp_path))
    assert obj.dimension == 3
    assert obj.model == "anything"


def test_factory_must_be_module_colon_attr() -> None:
    with pytest.raises(RuntimeError, match="module:attr"):
        _mcp_embed.build_embedder(_cfg(factory="no_colon_here"))


def test_factory_import_failure_is_clear() -> None:
    with pytest.raises(RuntimeError, match="failed to import"):
        _mcp_embed.build_embedder(_cfg(factory="nonexistent_pkg_xyz:build", model="m"))


def test_unknown_library_lists_known() -> None:
    with pytest.raises(RuntimeError, match="unknown embedder library 'frobnicate'"):
        _mcp_embed.build_embedder(_cfg(library="frobnicate", model="m"))


def test_missing_model_errors() -> None:
    with pytest.raises(RuntimeError, match="model is required"):
        _mcp_embed.build_embedder(_cfg(library="fastembed"))


def test_sentence_transformers_not_installed_message() -> None:
    # sentence-transformers is not a kglite dependency; the adapter must raise an
    # actionable "pip install sentence-transformers" rather than a bare ImportError.
    if "sentence_transformers" in sys.modules or _st_importable():
        pytest.skip("sentence-transformers is installed in this env")
    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        _mcp_embed.build_embedder(_cfg(library="sentence-transformers", model="BAAI/bge-m3"))


def _st_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None
