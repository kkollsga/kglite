"""`python -m kglite` launches the interactive shell (same as the `kglite`
console script)."""

from __future__ import annotations

import sys

from kglite.cli import main

if __name__ == "__main__":
    sys.exit(main())
