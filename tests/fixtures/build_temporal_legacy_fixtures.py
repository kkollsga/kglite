"""Regenerate the committed legacy temporal-configuration `.kgl` fixtures.

Before declarations got their own `temporal_declarations` metadata key, a
`.kgl` recorded validity bounds only in the `temporal_node_configs` /
`temporal_edge_configs` keys. A current build still reads those keys when the
new one is absent, and still writes them for closed configurations without
a source type, so an older binary keeps filtering. Asserting either direction
against files this tree wrote would be circular, so these fixtures are
produced by the **published 0.18.1 wheel** in an isolated interpreter, and
committed as binary.

Run this only to regenerate them (a fixture that no longer loads is a finding,
not a regeneration prompt). Run it from outside the repository root, so the
local `kglite/` package cannot shadow the installed wheel:

    uv venv /tmp/v0181 --python 3.12
    uv pip install --python /tmp/v0181/bin/python 'kglite==0.18.1' pandas
    cd /tmp && /tmp/v0181/bin/python <repo>/tests/fixtures/build_temporal_legacy_fixtures.py

What it writes under `tests/fixtures/temporal_legacy/`:

* `closed.kgl` — closed configs only, one per type: a `Field` node config and
  a `HAS_LICENSEE` relationship config, both from `validFrom`/`validTo`
  column types. Its two legacy keys are the bytes a current build must still
  write for the same graph.
* `identical_duplicates.kgl` — `HAS_LICENSEE` loaded twice with the same
  bound columns; 0.18.1 appended a config per load, so the list holds the
  same config twice.
* `distinct_duplicates.kgl` — `set_temporal` called twice on `HAS_LICENSEE`
  with different properties, the multi-config form 0.18.1 accepted without
  saying which source each belongs to.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import struct
import sys

import pandas as pd

import kglite

WHEEL = "0.18.1"
OUT = Path(__file__).resolve().parent / "temporal_legacy"


def _metadata(path: Path) -> dict:
    raw = path.read_bytes()
    (length,) = struct.unpack_from("<I", raw, 9)
    return json.loads(raw[13 : 13 + length])


def _base() -> "kglite.KnowledgeGraph":
    g = kglite.KnowledgeGraph()
    fields = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alpha", "Beta"],
            "vf": [dt.date(2000, 1, 1), dt.date(2005, 1, 1)],
            "vt": [dt.date(2010, 12, 31), None],
        }
    )
    g.add_nodes(fields, "Field", "id", "name", column_types={"vf": "validFrom", "vt": "validTo"})
    companies = pd.DataFrame({"id": [10, 11], "name": ["Acme", "Globex"]})
    g.add_nodes(companies, "Company", "id", "name")
    return g


def _licences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field": [1, 2],
            "company": [10, 11],
            "lic_from": pd.to_datetime(["2001-01-01", "2006-01-01"]),
            "lic_to": pd.to_datetime(["2003-12-31", None]),
        }
    )


def _connect(g: "kglite.KnowledgeGraph", column_types: dict | None) -> None:
    g.add_connections(
        _licences(),
        "HAS_LICENSEE",
        "Field",
        "field",
        "Company",
        "company",
        column_types=column_types,
    )


def main() -> None:
    if kglite.__version__ != WHEEL:
        sys.exit(f"refusing: these fixtures are written by kglite {WHEEL}, not {kglite.__version__}")
    if Path(kglite.__file__).resolve().is_relative_to(OUT.parents[1]):
        sys.exit("refusing: the repository's own kglite package shadows the wheel; run from outside it")
    OUT.mkdir(exist_ok=True)
    bounds = {"lic_from": "validFrom", "lic_to": "validTo"}

    closed = _base()
    _connect(closed, bounds)
    closed.save(str(OUT / "closed.kgl"))

    identical = _base()
    _connect(identical, bounds)
    _connect(identical, bounds)
    identical.save(str(OUT / "identical_duplicates.kgl"))

    distinct = _base()
    _connect(distinct, None)
    distinct.set_temporal("HAS_LICENSEE", "lic_from", "lic_to")
    distinct.set_temporal("HAS_LICENSEE", "other_from", "other_to")
    distinct.save(str(OUT / "distinct_duplicates.kgl"))

    edges = {
        name: _metadata(OUT / f"{name}.kgl")["temporal_edge_configs"]["HAS_LICENSEE"]
        for name in ("closed", "identical_duplicates", "distinct_duplicates")
    }
    assert len(edges["closed"]) == 1, edges
    same = edges["identical_duplicates"]
    assert len(same) == 2 and same[0] == same[1], edges
    distinct = edges["distinct_duplicates"]
    assert len(distinct) == 2 and distinct[0] != distinct[1], edges
    print(json.dumps(edges, indent=1))


if __name__ == "__main__":
    main()
