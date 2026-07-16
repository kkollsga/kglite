"""Progress reporters for ``KnowledgeGraph.load_ntriples``.

The Rust loader fires a structured event into a Python callable at
phase boundaries and within Phase 1's streaming loop. Pass any
callable to ``progress=`` to receive those events; this module ships
a tqdm-backed implementation for the common case.

Event schema (passed as a single dict argument):

  - ``{"kind": "start",    "phase": <str>, "label": <str>,
       "total": Optional[int], "unit": <str>}``
  - ``{"kind": "update",   "phase": <str>, "current": <int>, **counters}``
  - ``{"kind": "complete", "phase": <str>, "elapsed_s": <float>, **counters}``

Phases are ``"phase1"``, ``"phase1b"``, ``"phase2"``, ``"phase3"``,
``"finalising"`` (some are skipped depending on storage mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["TqdmBuildProgress"]

if TYPE_CHECKING:
    from typing import Any as _Any


class TqdmBuildProgress:
    """Drop-in tqdm-backed progress reporter for ``load_ntriples``.

    One bar per phase, transitioning as `start`/`complete` events
    arrive. `update` events drive the bar position and write counters
    + RSS into the postfix.

    Requires ``tqdm`` (and ``psutil`` if ``memory=True``); both are
    imported lazily so kglite itself does not depend on them.

    Usage::

        from kglite import KnowledgeGraph
        from kglite.progress import TqdmBuildProgress
        g = KnowledgeGraph()
        g.load_ntriples(path, progress=TqdmBuildProgress())
    """

    def __init__(self, memory: bool = True) -> None:
        try:
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError("TqdmBuildProgress needs tqdm — `pip install tqdm`") from e
        self._tqdm = tqdm
        self._bar: _Any = None
        self._proc: _Any = None
        if memory:
            try:
                import psutil

                self._proc = psutil.Process()
            except ImportError as e:
                raise ImportError(
                    "TqdmBuildProgress(memory=True) needs psutil — `pip install psutil`, or pass memory=False"
                ) from e

    def _build_postfix(self) -> dict:
        """Tight postfix: just `mem`. The bar's auto-rate (`X.XMtri/s`,
        `X.XMedge/s`) carries the throughput info; per-event counters
        ride along on the raw event for any UI that wants them, but the
        terminal bar stays scannable."""
        if self._proc is None:
            return {}
        return {"mem": f"{self._proc.memory_info().rss / 1e9:.1f}GB"}

    def __call__(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "start":
            if self._bar is not None:
                self._bar.close()
            self._bar = self._tqdm(
                total=event.get("total"),
                desc=event.get("label", event.get("phase", "")),
                unit=event.get("unit", "it"),
                unit_scale=True,
                dynamic_ncols=True,
            )
        elif kind == "update":
            if self._bar is None:
                return
            cur = event.get("current")
            if cur is not None:
                # Set absolute position rather than incrementing — Rust
                # already gives us the running total.
                self._bar.n = cur
            self._bar.set_postfix(self._build_postfix(), refresh=False)
            self._bar.refresh()
        elif kind == "complete":
            if self._bar is None:
                return
            # Phases that don't fire per-update events (1b/2/3/finalising)
            # would otherwise close at n=0 — bump to total so the final
            # rendered line shows 100%, not 0% of a known-total bar.
            if self._bar.total is not None and self._bar.n < self._bar.total:
                self._bar.n = self._bar.total
            self._bar.set_postfix(self._build_postfix(), refresh=False)
            self._bar.refresh()
            self._bar.close()
            self._bar = None

    def close(self) -> None:
        """Idempotent cleanup. Useful if the load aborts before its
        final ``complete`` event fires."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None
