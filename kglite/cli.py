"""Console-script shim for the Rust ``kglite`` CLI bundled in the wheel.

The CLI implementation lives in the ``kglite-cli`` Rust library and is linked
into this wheel's existing extension alongside the graph engine. This module
only forwards argv and formats a top-level error; command parsing and behavior
are shared with the standalone ``kglite-cli`` binary distribution.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the bundled Rust CLI with ``argv`` or ``sys.argv[1:]``."""
    from kglite import _run_cli

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        _run_cli(args)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(f"kglite: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
