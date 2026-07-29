"""Baseline: time building the Sodir graph from blueprint.

Runs preprocess + from_blueprint() a few times and reports median.
Used as before-number for Rust blueprint port.
"""

import contextlib
import io
from pathlib import Path
import statistics
import time

import kglite

SCRIPT_DIR = Path(__file__).parent
BLUEPRINT = str(SCRIPT_DIR / "sodir_graph_config.json")

# Import preprocess_csvs from the existing benchmark
import sys

sys.path.insert(0, str(SCRIPT_DIR))
from benchmark_sodir import preprocess_csvs  # noqa: E402


def main(iterations: int = 3, warmup: int = 1) -> None:
    print(f"KGLite v{kglite.__version__} — blueprint build baseline")
    print(f"Blueprint: {BLUEPRINT}")
    print()

    with contextlib.redirect_stdout(io.StringIO()):
        preprocess_csvs()

    times_ms: list[float] = []
    for i in range(warmup + iterations):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            graph = kglite.from_blueprint(BLUEPRINT, save=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        label = "warmup" if i < warmup else f"run{i - warmup + 1}"
        print(f"  {label}: {elapsed_ms:>9.1f} ms")
        if i >= warmup:
            times_ms.append(elapsed_ms)

    s = graph.schema()
    print()
    print(f"  nodes: {s['node_count']:,}")
    print(f"  edges: {s['edge_count']:,}")
    print()
    print(f"  median: {statistics.median(times_ms):>9.1f} ms")
    print(f"  min:    {min(times_ms):>9.1f} ms")
    print(f"  max:    {max(times_ms):>9.1f} ms")


if __name__ == "__main__":
    main()
