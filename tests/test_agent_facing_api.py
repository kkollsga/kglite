"""Parity gate for the deliberately cross-surface, agent-facing API.

This checks names and ownership only. Argument decoding, result marshalling,
display, and transport errors intentionally remain binding-specific.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "api-baselines" / "agent-facing.json"
SURFACES = {"core", "python_stub", "introspection", "mcp"}


def test_agent_facing_manifest_has_every_required_surface() -> None:
    entries = json.loads(MANIFEST.read_text())
    assert entries, "agent-facing API manifest must not be empty"
    for name, surfaces in entries.items():
        assert set(surfaces) == SURFACES, f"{name}: incomplete surface declaration"


def test_agent_facing_symbols_exist_on_every_declared_surface() -> None:
    entries = json.loads(MANIFEST.read_text())
    for name, surfaces in entries.items():
        for surface, (relative_path, token) in surfaces.items():
            path = ROOT / relative_path
            assert path.is_file(), f"{name}/{surface}: missing {relative_path}"
            assert token in path.read_text(), (
                f"{name}/{surface}: {token!r} absent from {relative_path}; "
                "update the implementation and manifest together"
            )
