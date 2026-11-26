"""``export_rdf`` streams: the footprint of exporting a 600 k-row type does not grow
with the size of the output.

The export runs in a child process whose resident size the parent samples, so
the measurement does not depend on the export releasing the GIL. The N-Quads
written exceed 150 MB; a writer that held the statements (or a type's text)
would add at least that much, while the streaming writer adds a few MB.
Only Darwin asserts the ceiling (the child's RSS on other
platforms includes allocator behaviour this test has not been calibrated on).
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

psutil = pytest.importorskip("psutil")

ROWS = 600_000
CEILING_MB = 50

CHILD = textwrap.dedent(
    """
    import sys, time
    import numpy as np, pandas as pd
    from kglite import KnowledgeGraph

    rows = int(sys.argv[2])
    g = KnowledgeGraph()
    df = pd.DataFrame({
        "id": np.arange(rows),
        "name": [f"person-{i:09d}-" + "x" * 60 for i in range(rows)],
        "salary": np.arange(rows, dtype="float64") * 0.5,
    })
    g.add_nodes(df, "Person", "id", "name")
    del df
    print("READY", flush=True)
    time.sleep(1.5)
    summary = g.export_rdf(sys.argv[1])
    print("DONE", summary["nodes"]["Person"], flush=True)
    time.sleep(1.0)
    """
)


@pytest.mark.skipif(sys.platform != "darwin", reason="footprint ceiling is calibrated on Darwin")
def test_rdf_export_footprint_is_bounded(tmp_path):
    env = dict(os.environ)
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(tmp_path / "out.nq"), str(ROWS)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    proc = psutil.Process(child.pid)
    try:
        assert child.stdout.readline().strip() == "READY"
        time.sleep(0.5)
        baseline = proc.memory_info().rss
        peak = baseline
        done = None
        deadline = time.time() + 100
        while time.time() < deadline and done is None:
            peak = max(peak, proc.memory_info().rss)
            if child.poll() is not None:
                break
            time.sleep(0.01)
            # DONE is printed after the export returns; stop sampling then.
            import select

            if select.select([child.stdout], [], [], 0)[0]:
                done = child.stdout.readline().strip()
        assert done == f"DONE {ROWS}", done
    finally:
        child.kill()
        child.wait()
    delta_mb = (peak - baseline) / 1e6
    size_mb = (tmp_path / "out.nq").stat().st_size / 1e6
    assert size_mb > 150, f"fixture too small to prove anything: {size_mb:.0f} MB"
    print(f"export delta {delta_mb:.1f} MB for {size_mb:.0f} MB of N-Quads")
    assert delta_mb < CEILING_MB, f"export grew RSS by {delta_mb:.0f} MB writing {size_mb:.0f} MB"
