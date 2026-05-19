"""SEC EDGAR dataset loader for KGLite.

Public API:

    from kglite.datasets.sec import SEC
    g = SEC.open(path, *, years=10, detailed=2, mode="mapped",
                 user_agent="Name email@dom")

The workdir holds three tiers (raw/, processed/, graph/{mode}/). See the
class docstring on ``SEC`` for the full lifecycle.
"""

from .wrapper import SEC

__all__ = ["SEC"]
