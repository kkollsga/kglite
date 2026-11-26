"""``export_csv`` streams: the footprint of exporting a 1 M-row type does not grow
with the size of the output.

The export runs in a child process whose resident size the parent samples, so
the measurement does not depend on the export releasing the GIL. The CSV
written is ~95 MB; a writer that held a type's text (the old one built one
``String`` per type) would add at least that much, while the streaming
writer adds a few MB. Only Darwin asserts the ceiling (the child's RSS on other
platforms includes allocator behaviour this test has not been calibrated on).

The meter is resident size, and a resident-size baseline taken while part of
the graph is swapped out or compressed reads the pages the export then faults
back in as export growth (100+ MB against ~3 MB when idle, seen under 4.8 GB of
swap). So the child reads every column the export reads (a scan, not an
export: a warm-up export would leave the allocator holding the very memory a
materialising writer needs, and hide it) before it reports ready, and the
baseline is the highest resident size over a settle window rather than one
sample. Footprint (``phys_footprint``) is no substitute: it charges compressed
pages at their compressed size, and a whole-type ``String`` of CSV text
compresses ~3x, which kept a deliberately materialising writer under the
ceiling.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

psutil = pytest.importorskip("psutil")

ROWS = 1_000_000
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
    # Fault in every column the export reads (id, title, salary) before the
    # parent takes its baseline.
    g.cypher("MATCH (p:Person) RETURN sum(p.id) AS i, sum(size(p.title)) AS t, sum(p.salary) AS s").to_list()
    print("READY", flush=True)
    time.sleep(1.5)
    summary = g.export_csv(sys.argv[1])
    print("DONE", summary["nodes"]["Person"], flush=True)
    time.sleep(1.0)
    """
)


@pytest.mark.skipif(sys.platform != "darwin", reason="footprint ceiling is calibrated on Darwin")
def test_export_footprint_is_bounded(tmp_path):
    env = dict(os.environ)
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(tmp_path / "out"), str(ROWS)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    proc = psutil.Process(child.pid)
    try:
        assert child.stdout.readline().strip() == "READY"
        baseline = 0
        settle_until = time.time() + 1.0
        while time.time() < settle_until:
            baseline = max(baseline, proc.memory_info().rss)
            time.sleep(0.02)
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
    size_mb = (tmp_path / "out" / "nodes" / "Person.csv").stat().st_size / 1e6
    assert size_mb > 80, f"fixture too small to prove anything: {size_mb:.0f} MB"
    print(f"export delta {delta_mb:.1f} MB for {size_mb:.0f} MB of CSV")
    assert delta_mb < CEILING_MB, f"export grew RSS by {delta_mb:.0f} MB writing {size_mb:.0f} MB"
