"""KGLite dataset helpers — opinionated builders for well-known public
datasets. Each submodule wraps the fetch + maintenance + build cycle
behind a single entry point so applications can treat a public dataset
as a typed Python value.

Submodules:
    sec      - SEC EDGAR filings (pure-Rust loader, no pandas).
    sodir    - Norwegian Continental Shelf petroleum data.
    wikidata - Wikimedia Foundation's `latest-truthy` RDF dumps.

Submodules load lazily (PEP 562). `import kglite.datasets.sec` must
not drag in `sodir`/`wikidata` and their pandas/pyarrow stack: that
stack is unrelated to SEC, slows startup, and — on macOS — loading
pyarrow after the kglite native extension triggers a dynamic-linker
crash in the extension. Accessing a submodule by name
(`kglite.datasets.sodir`) imports it on first use.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["sec", "sodir", "wikidata"]

if TYPE_CHECKING:
    from . import sec, sodir, wikidata  # noqa: F401


def __getattr__(name: str) -> object:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
