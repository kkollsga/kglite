"""Console-script launcher for the bundled ``kglite`` interactive shell.

``pip install kglite`` ships the compiled ``kglite`` shell binary as package
data (under ``kglite/_bin/``); this thin launcher execs it as a real
subprocess so the REPL keeps full terminal control (raw mode, signal
handling) — none of the rustyline-inside-the-extension fragility.

If the binary is not bundled for the current platform (e.g. a target whose
cross-compile isn't wired up yet), it prints a clear fallback pointing at
``cargo install kglite-cli``.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

_FALLBACK = (
    "kglite: the interactive shell binary is not bundled in this wheel for your "
    "platform.\n"
    "Install it with Rust instead: `cargo install kglite-cli` (provides the "
    "`kglite` command),\n"
    "or run queries from Python via `import kglite`.\n"
)


def _binary_path() -> Path:
    """Location of the bundled shell binary inside the installed package."""
    name = "kglite.exe" if os.name == "nt" else "kglite"
    return Path(__file__).resolve().parent / "_bin" / name


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``kglite`` console script. Forwards argv to the
    bundled binary and returns its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    binary = _binary_path()
    if not binary.exists():
        sys.stderr.write(_FALLBACK)
        return 1
    try:
        return subprocess.run([str(binary), *args], check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
