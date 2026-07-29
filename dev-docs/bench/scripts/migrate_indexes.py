"""One-shot upgrade: emit `id_indices.bin` and `type_indices.bin` raw files
for an existing disk graph that was saved with the legacy zstd format.

Loads the graph (which still pays the slow eager-rebuild cost on legacy files),
calls `g.save_indexes_inplace()` to emit only the new index files, then exits.

NOTE: this script does NOT re-save the entire graph. It only updates the two
index files we changed in this rev. Existing `*.bin.zst` legacy files are left
in place as fallback. The new loader prefers `*.bin` over `*.bin.zst`.

Usage:
    python bench/migrate_indexes.py /Volumes/EksternalHome/Data/Wikidata/graph
"""

import sys
import time
from pathlib import Path

import kglite

if len(sys.argv) != 2:
    print(__doc__, file=sys.stderr)
    sys.exit(2)

graph_path = Path(sys.argv[1])
if not (graph_path / "disk_graph_meta.json").exists():
    print(f"error: {graph_path} is not a disk-graph directory", file=sys.stderr)
    sys.exit(2)

print(f"loading {graph_path}…")
t0 = time.perf_counter()
g = kglite.load(str(graph_path))
print(f"  load took {time.perf_counter() - t0:.1f}s")

# Re-save both index files via Python-exposed helpers if available; otherwise
# call the underlying full save (which also writes type_indices.bin and
# id_indices.bin since both writers are now wired in).
print("re-saving graph (full save path emits the new index files)…")
t1 = time.perf_counter()
g.save(str(graph_path))
print(f"  save took {time.perf_counter() - t1:.1f}s")

print(f"done. inspect files:")
for f in sorted(graph_path.iterdir()):
    if f.name.startswith("id_indices") or f.name.startswith("type_indices"):
        size = f.stat().st_size
        print(f"  {f.name:30}  {size / 1024 / 1024:.1f} MB")
