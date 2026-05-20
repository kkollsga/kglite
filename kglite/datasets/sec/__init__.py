"""SEC EDGAR dataset loader for KGLite.

Public API:

    from kglite.datasets.sec import SEC

    # Ergonomic shortcut — name a form, a company, a span:
    g = SEC.fetch(path, "13F-HR", "TSLA", years=2,
                  user_agent="Name email@dom")

    # Full control — separate index/detail spans, storage mode, flags:
    g = SEC.open(path, *, years=10, detailed=2, mode="mapped",
                 user_agent="Name email@dom")

The workdir holds three tiers (raw/, processed/, graph/{mode}/). See the
class docstring on ``SEC`` for the full lifecycle.
"""

from .wrapper import SEC

__all__ = ["SEC"]
