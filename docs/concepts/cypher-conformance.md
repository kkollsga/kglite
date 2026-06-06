# Cypher Conformance — On-Demand Check Against Neo4j

KGLite ships a focused openCypher subset, not a Neo4j drop-in
replacement. The regular test suite (CI) checks self-consistency:
optimizer-on vs optimizer-off (the differential corpus), and memory
vs mapped vs disk (the parity oracles). Neither of those catches a
bug where *every* code path is wrong in the same way.

`scripts/cypher_conformance.py` is the absolute-correctness oracle —
on-demand, opt-in, never wired into pytest or CI. When you suspect a
semantic divergence (NULL handling, aggregate corner case, path
operator behaviour), bring up Neo4j locally, run the script, get a
PASS/FAIL report.

## When to run it

- After a non-trivial change to the Cypher executor (`src/graph/languages/cypher/executor/`).
- When triaging a "Neo4j gives X but KGLite gives Y" bug report.
- Before claiming any specific Cypher behaviour is "openCypher-compliant" in docs or PRs.
- Periodically — once per major release is a reasonable rhythm.

Not for:

- Every PR. The Docker setup + Neo4j boot + per-fixture push adds
  ~30 seconds to a check that the regular suite already covers from
  a self-consistency angle.
- Catching planner regressions. That's the differential corpus's job
  (`tests/test_cypher_differential.py`).

## How it works

The runner reuses the existing differential corpus and the shared
pytest fixtures:

1. For each `(name, fixture, query, params)` in `DIFFERENTIAL_QUERIES`,
   the script builds the fixture *once* via the `build_*` helpers in
   `tests/conftest.py` and pushes it to Neo4j via the existing
   `kglite.to_neo4j()` export.
2. The query runs on both KGLite and Neo4j.
3. Results are normalised (row order ignored unless `ORDER BY` is
   present) and compared.
4. A summary plus per-failure detail (query text + first 5 rows from
   each side + set diff) is printed.

The same corpus is the per-PR differential gate's input, so both
engines see identical queries.

## Workflow

```bash
make neo4j-up               # bring up Neo4j 5 in a container
make neo4j-conformance      # run the script
make neo4j-down             # tear down
```

The `neo4j-conformance` target installs the optional `[neo4j]` extras
(`pip install -e '.[neo4j]'`) on first run.

Useful flags (pass via `python scripts/cypher_conformance.py …`):

- `--filter SUBSTRING` — only run queries whose name contains the
  substring. Handy when iterating on a single divergence.
- `--verbose` — print every query, not just failures.

Exit code: 0 if every query passes (modulo intentional divergences);
1 otherwise.

## Interpreting output

```
  PASS   simple_match                            (5 rows)
  SKIP   div_by_zero                             (intentional: KGLite returns NULL for x/0)
  SKIP   wikidata_cohort                         (KGLite-only feature)
  FAIL   ne_with_null                            kglite=3 rows, neo4j=5 rows
```

- `PASS` — row sets match. Either both are correct, or both wrong in
  the same way (rare — the differential corpus would have caught
  most "wrong in the same way" cases at PR time).
- `SKIP (intentional)` — entry registered in
  `INTENTIONAL_DIVERGENCES` with a one-line rationale.
- `SKIP (KGLite-only)` — query references a KGLite-only feature
  (e.g. `CALL kglite.affected_tests(...)`, `text_score(...)`,
  `FORMAT CSV`). Skipped because Neo4j has no equivalent.
- `FAIL` — real divergence to investigate.

For each failure, the script prints both row sets (first 5) and a
set diff highlighting rows present in only one side.

## Registering a deliberate divergence

When investigation concludes that KGLite's behaviour is intentionally
different from Neo4j — and spec-defensible (e.g. integer division
returns int per the 0.9.0 §5 fix; division by zero returns NULL
rather than NaN/Inf per the 0.9.52 numeric-boundaries pin) — add an
entry to `INTENTIONAL_DIVERGENCES` in
`scripts/cypher_conformance.py`:

```python
INTENTIONAL_DIVERGENCES: dict[str, str] = {
    "div_by_zero": "KGLite returns NULL for x/0 (int and float); Neo4j returns NaN/Infinity for float — see CHANGELOG 0.9.52",
}
```

Entries should be the exception, not the rule. **Most "Neo4j differs"
findings are bugs to fix, not divergences to register.** The point
of the divergence registry is to keep the gate honest — it stops
flagging the cases that are deliberate, so the cases that aren't
stand out.

## Bolt wire conformance

There's a second, sibling runner — `scripts/bolt_conformance.py` — with a
different oracle. Instead of diffing KGLite against Neo4j (the *spec*
oracle), it diffs the **Bolt wire path against direct in-process
`cypher()`**. Both sides run the same engine, so any divergence is a
PackStream / wire round-trip bug in `kglite-bolt-server`, not a semantic
one. Because the oracle is KGLite-in-process, **no Neo4j or Docker is
needed** — the runner spawns its own `kglite-bolt-server` on an ephemeral
port (reusing the launch helpers in `tests/conftest.py`), runs each corpus
query over the `neo4j` Python driver, and compares against
`KnowledgeGraph.cypher()`.

```bash
make bolt-conformance       # builds the binary, spawns it, runs the corpus
```

The target builds `kglite-bolt-server --release` and installs the `[neo4j]`
extra (for the driver) on first run. Flags mirror the Neo4j runner:
`--filter SUBSTRING`, `--verbose`. Exit code 0 if every query round-trips
identically; 1 on any divergence.

This is the on-demand companion to `tests/test_bolt_server_differential.py`,
which runs the same corpus over the wire as part of the `-m bolt` suite —
the script is for ad-hoc investigation when a wire-encoding bug is
suspected.

## Why this isn't in the test suite

Per the test-suite-fortification design (see
`docs/concepts/design-decisions.md` — Test gates section, *if added*):
the project's stated principle is that the regular test run depends
only on what `pip install -e .[mcp]` provides. Wiring Neo4j into the
suite would mean:

- CI needs a Docker layer or a running Neo4j service.
- Local `pytest tests/` requires Docker installed.
- Network flakes (driver timeouts, container startup races) surface
  as flaky test failures.
- The default `pytest tests/` runtime climbs by ~30 seconds.

None of those are worth paying for a check that you only need a few
times per release. On-demand and opt-in lets the runner be a
deliberate tool the maintainer reaches for, not a tax on every
contributor's commit cycle.
