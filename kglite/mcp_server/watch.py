"""File watcher — thin wrapper around the mcp-methods Rust debouncer.

0.9.24: replaced the pure-Python `watchdog` + threading debouncer with
a single call into `kglite._mcp_internal.start_watch`. Rust drives the
`notify-debouncer-mini` event loop on a background thread; the
callback runs with the GIL re-acquired automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from kglite import _mcp_internal

log = logging.getLogger("kglite.mcp_server.watch")


def start(
    dir_path: Path,
    on_change: Callable[[list[str]], None],
    debounce_seconds: float = 0.5,
) -> Any:
    """Watch `dir_path` recursively; call `on_change(paths)` after
    each debounce window. `paths` is the list of changed file paths
    (post-debounce) — the caller can filter to only fire its actual
    work on graph-relevant subsets (e.g. via
    `kglite._kglite_code_tree.language_for_path`).

    Returns a `WatchHandle` whose `.stop()` tears the watcher down
    — caller must keep a reference (the handle's Drop in Rust is
    what unregisters the watcher).

    2026-05-25: signature changed to pass paths through; previously
    `on_change()` was no-arg and the wrapper discarded paths.
    """

    def _dispatch(paths: list[str]) -> None:
        try:
            on_change(paths)
        except Exception as e:
            log.warning("watch on_change handler raised: %s", e)

    handle = _mcp_internal.start_watch(
        str(dir_path),
        _dispatch,
        debounce_ms=int(debounce_seconds * 1000),
    )
    log.info("watching %s (debounce=%.2fs)", dir_path, debounce_seconds)
    return handle
