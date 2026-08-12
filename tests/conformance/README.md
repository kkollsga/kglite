# Official Bolt driver conformance suites

`kglite-bolt-server` speaks the Bolt v5 wire protocol, which means the official
Bolt drivers should work against it unmodified. "Should" is not a test. These
suites make it one for the two non-Python drivers that matter most:

| Suite | Driver | Runs |
|---|---|---|
| [`js/`](js) | `neo4j-driver` (npm) | anywhere `node` + `npm` exist |
| [`java/`](java) | `org.neo4j.driver:neo4j-java-driver` (Maven) | needs a JDK 17+ and Maven |

The official **Python** driver is covered separately and more broadly, by
`tests/test_bolt_server_*.py` and `scripts/bolt_conformance.py`.

## Why a second and third driver at all

Each official driver ships its own PackStream codec, its own managed-transaction
retry machinery, and its own exception hierarchy. A server can satisfy one
driver's expectations and violate another's — the Python suite cannot detect
that by construction. This is not hypothetical: writing these suites is what
surfaced that OCC commit conflicts were reporting
`Neo.ClientError.Transaction.TransactionStartFailed` while the README promised
a different code. The Python tests matched on message text and never noticed; a
driver-idiomatic retry loop branches on the code, so it would have. Conflicts
now report `Neo.TransientError.Transaction.Outdated`, and each driver's
retry machinery keys off that class prefix — so these suites are also where a
driver that declines to retry it would show up.

**Java specifically.** The JVM has no in-process route to a kglite graph — no
JNI binding, no embedded engine — so the Bolt server *is* the integration
surface for JVM consumers. That makes the Java driver the one whose behaviour
we can least afford to assume, which is why it is verified.

## Running them

Both are driven by one pytest module, which builds the server, starts it on an
ephemeral port, hands over the URI, and asserts the two suites stayed in
parity:

```bash
cargo build --release -p kglite-bolt-server      # or: make build-bolt-server
pytest tests/test_bolt_driver_conformance.py -m bolt -v -rs
```

`-m bolt` is required — the marker is deselected by default in
`pyproject.toml`'s `addopts`. `-rs` prints skip reasons, which is how you tell
"the toolchain is missing" from "it ran".

Standalone, against an already-running server:

```bash
cd tests/conformance/js   && npm install && node conformance.mjs bolt://127.0.0.1:7687
cd tests/conformance/java && mvn -B test -Dkglite.bolt.uri=bolt://127.0.0.1:7687
```

## Keeping them in parity

Both suites cover the **same named checks**, and
`test_both_suites_cover_the_same_checks` enforces it by reading the sources —
so parity is checked even on a machine with neither runtime installed. Adding a
check means adding it to both suites and bumping `EXPECTED_CHECKS` in
`tests/test_bolt_driver_conformance.py`; that constant also catches a suite that
silently stops halfway rather than reporting a cheerful green.

Check names, grouped:

- `connectivity.verify`
- `session.scalar_return`, `session.parameters`
- `types.{integer,float,string,boolean,null,list,map}`
- `graph.{node,relationship,path}`
- `tx.{explicit_write_commits,rollback_discards,executeWrite_managed_retry,autocommit_mutation_is_rejected,occ_conflict_code}`
- `errors.{syntax_error_code,codes_are_neo4j_shaped}`
- `procedures.db_labels`
- `capability.load_csv_denied_for_remote_clients`

The last one is a security assertion, not a feature one: the server under test
is started **without** `--allow-csv-import`, so a remote `LOAD CSV` must be
refused. See `crates/kglite/src/graph/languages/cypher/executor/load_csv.rs`.

## Artifacts

`js/node_modules`, `java/target`, and `java/.m2` are fetched dependencies and
build output. They are gitignored and bounded by `make prune-dev`, per the
dev-cleanliness rule that every path the tooling writes outside git needs an
owner.
