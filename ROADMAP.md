# KGLite Roadmap

> Strategy doc. The day-to-day fix list lives in `CHANGELOG.md`. This file
> answers "what kind of thing is KGLite trying to be, and what comes next."

## Vision

KGLite is the **embedded openCypher engine for LLM-agent workloads**: a
graph database you `pip install`, hand to an agent via MCP, and forget
about. The Python-native embedded story is the heart; everything else
hangs off it.

We're not trying to be Neo4j (server, multi-process, distributed). We're
not trying to be DuckDB-PGQ (columnar OLAP). We're the thing that sits
inside your agent's process, exposes a Cypher surface and an
introspection-shaped schema, and gets out of the way.

---

## Positioning vs Kuzu

KuzuDB was the closest direct competitor: embedded property-graph, Cypher,
multi-language. It was archived in 2025. The seat for "the embedded
openCypher engine" is currently empty. KGLite should take it — not by
out-Kuzuing Kuzu on what Kuzu was good at, but by leaning into the
adjacent space Kuzu never owned: the LLM-agent surface.

| Axis | Kuzu was strong at | KGLite is strong at | Strategic call |
|---|---|---|---|
| **Cypher coverage** | Closer to Neo4j parity | openCypher subset, recently fortified for correctness | **Close the gap incrementally** — coverage that real consumers ask for, not spec maximalism. |
| **Storage / scan perf** | Columnar layout, vectorized execution, multi-GB analytical workloads | In-memory wins; mapped + disk are addons | **Don't compete head-on.** Stay honest: in-memory is the design centre. |
| **Multi-language bindings** | C++, Python, Java, Node, Rust, .NET | Python only (PyO3 + abi3) | **Catch up — but only the bindings agents actually use** (Node first, then JVM). |
| **LLM / agent surface** | None | MCP server bundled, `describe()` for system prompts, code-tree parser, domain dataset loaders | **Lean into this hard.** No competitor in the space has it. |
| **Domain datasets** | None | SEC, Sodir, Wikidata, code-tree built-in | **Lean in selectively** — see "Tone down" below. |
| **Wire protocol** | Embedded only | Embedded only | **Add Bolt.** The single biggest move (see Roadmap §1). |

---

## Identity — what to lean into, what to tone down

### Lean in

- **Embedded-first.** "Single-process, in-memory wins" is a design choice,
  not a limitation. It's why agents can hand off the whole graph as a
  `.kgl` file and continue elsewhere.
- **LLM-agent surface.** MCP server, `describe()`, the manifest-driven
  tool-registration story, the `kglite-mcp-server` binary. No competitor
  has this. Every new feature should ask "does this make agent UX
  better?"
- **Code-tree.** No other graph DB ships a multi-language source-code
  parser that produces a queryable graph. It's a unique answer to "give
  the agent your codebase as a knowledge graph."
- **Honest Cypher.** The 0.9.52 NULL-semantics fixes + the Phase-5
  on-demand Neo4j conformance runner say "we'll be correct first,
  fast second." This is a hard-won differentiator from Cypher engines
  that ship silent wrong rows.
- **One-file persistence.** `.kgl` files are a complete, portable,
  reproducible snapshot. SQLite's killer feature was this; we have the
  graph-shaped equivalent.

### Tone down — or keep, but de-emphasise

- **Sodir (Norwegian petroleum NCS) as a first-class README example.**
  It's a great dataset that demonstrates the model, but the README
  shouldn't lead with it for a general audience. Move to
  `docs/guides/datasets.html` as one of several examples; keep
  Wikidata + SEC + code-tree up top.
- **The fluent API as a parallel surface to Cypher.** Maintaining two
  query surfaces costs us. The fluent API is occasionally more ergonomic
  in Python, but most agent workloads now go through Cypher. Don't add
  features to the fluent API unless they're motivated by user pull;
  let Cypher be the primary surface.
- **Spatial / timeseries breadth.** Both pay binary-size cost (see the
  35→60 MB platform divergence we just discovered). Keep what's there;
  don't expand to a "full" spatial DB unless a real consumer asks.
- **The disk modes' weight.** Per `CLAUDE.md`: "in-memory wins every
  time." Disk modes are an addon for large-graph exploration (Wikidata
  scale). When optimisation conflicts arise, in-memory wins — already
  the rule. Just don't let the disk modes drag us toward a fundamentally
  different shape (e.g. don't add multi-process concurrency for disk
  alone; that's a different product).

### Explicitly NOT on the roadmap (and why)

| Won't do | Use this instead | Why |
|---|---|---|
| Multi-process write concurrency | Neo4j, Memgraph | "Embedded" implies single-process. Adding multi-process turns us into a worse Neo4j. |
| Horizontal sharding / distribution | Neptune, TigerGraph | Same: opposite of embedded. |
| Sub-microsecond OLTP point queries | Postgres, KeyDB | Wrong shape for a graph DB. |
| Vectorized columnar OLAP for billions of rows | DuckDB-PGQ, Kuzu's successor if one emerges | Different design centre. Stay focused. |
| Full GraphQL surface | A separate graphql-to-cypher layer | Out of scope for the core. |

---

## Roadmap

Ordered by leverage. Each item links to its own scoping doc when one
exists (or will be added when work starts).

### §1 — Bolt protocol server 🚧

> **Status:** 🚧 **In progress** — see [`bolt_implementation.md`](bolt_implementation.md)
> for the phased plan (Phase A core preparations → B test contract +
> perf baselines → C the protocol itself → D end-to-end test program
> + release). Each phase is planned, implemented, and committed in
> its own plan loop.

**Why.** Every existing Neo4j-aware tool — BloodHound, the Neo4j Browser,
LangChain's `Neo4jGraph`, llama-index, every Python/JS/Java/Go Neo4j
client — already knows how to talk to a graph DB via the Bolt binary
protocol. If KGLite speaks Bolt, the entire Neo4j ecosystem becomes
plug-and-play with no changes from the consumer side. This is the
single highest-leverage feature we could add.

**Scope (initial).** Bolt v5 server (TCP) supporting `BEGIN/RUN/COMMIT`,
parameter binding, result streaming, and the subset of types KGLite
already represents internally. Pure-Rust implementation living in a new
crate `crates/kglite-bolt-server`. Optional binary `kglite-bolt-server`
that opens a `.kgl` file (or `--memory`) and accepts Bolt connections.

**Out of scope (initial).** Authentication beyond basic password (no
Kerberos/LDAP/SSO), Neo4j-specific procedures (`db.labels()`, `db.indexes()`
etc. — we'd add a curated subset on demand), routing for clusters
(we're embedded — there is no cluster).

**Effort.** 3-5 weeks. Bolt is well-documented; the encoding is a small
PackStream variant. The Rust ecosystem has the framing primitives.

**Depends on.** Nothing in the kglite core. Lands as a sibling crate.

**Detailed scoping.** See [`BOLT.md`](BOLT.md) once that doc exists.

---

### §2 — Node.js binding

**Why.** Node/TS is where the LLM-tooling ecosystem lives (LangChain.js,
llama-index TS, Mastra, every browser-attached agent). A Python-only
graph DB cuts ourselves off from half the agent market.

**Scope.** napi-rs binding exposing the same surface as the PyO3
binding: `KnowledgeGraph`, `cypher()`, `add_nodes()`, fluent API,
`describe()`. Ships as `@kglite/kglite` npm package. Reuses the existing
Rust core unchanged.

**Out of scope.** A separate JS API surface — match Python's shape.

**Effort.** 1-2 weeks once Bolt lands (or independent of it).

**Depends on.** A stable Rust API at `kglite_core::*` (currently
intertwined with PyO3 in places; small refactor required).

---

### §3 — C ABI + headers

**Why.** Foundation for every non-Python, non-Node binding. Languages
without first-class Rust FFI (Java, Go, C#, Ruby) all bind through C.
Currently kglite has a C-shaped surface buried under PyO3; surfacing it
deliberately + producing `kglite.h` via cbindgen makes the door open.

**Scope.** `crates/kglite-capi` exposing the GraphRead/GraphWrite
subset, Cypher run + result iteration, save/load. Header generated
into `target/include/kglite.h` for downstream linking. CI builds it
across the matrix.

**Out of scope.** A full GraphIQ-style C++ wrapper. Stay minimal.

**Effort.** 1-2 weeks. Mostly mechanical.

**Depends on.** §2's Rust-API stabilisation makes this easier.

---

### §4 — Connection-string convention

**Why.** Every modern data tool expects `<scheme>://<path>` configuration.
Bolt already gives us `bolt://localhost:7687`. For embedded use we
should also have `kglite://./graphs/mygraph.kgl` consistently — that's
how tools wire up DSNs.

**Scope.** Standardise `kglite://` in the README, docs, and any
constructor that takes a path. Add a `KnowledgeGraph.from_url()`
constructor that accepts both `kglite://` (embedded) and `bolt://`
(client mode — needs §1).

**Out of scope.** Parsing every variant. Just the two.

**Effort.** Days.

**Depends on.** §1 (for the `bolt://` half).

---

### §5 — Multi-label nodes (Track C from the rationale doc)

**Why.** A real openCypher engine needs multi-label. BloodHound,
Wikidata-style "X is a Human AND a Politician", and Bolt-using
consumers all assume it. `docs/explanation/multi-label-rationale.md`
documents the current single-label design choice and what a future
implementation would look like.

**Scope.** Per the rationale doc's "stepping-stone" path: add
`Value::List` first (independently valuable), then
`NodeData.secondary_labels: SmallVec<InternedKey>` + a graph-level
`has_secondary` flag so single-label graphs cost nothing. Keep primary
as the columnar grouping key (no schema union). Reference: mrmagooey's
`multi-label` fork branch already prototypes most of this on an older
kglite — worth porting forward.

**Out of scope.** Restructuring v3 columnar layout.

**Effort.** 3-4 weeks. The rationale doc estimates XL; the additive
stepping-stone path keeps it tractable.

**Depends on.** Decision to commit (currently deferred — see the
rationale doc). The Bolt work (§1) may force the decision: if
Bolt-attached tools break on single-label nodes, multi-label moves
from "nice to have" to "required for §1 to be useful."

---

### §6 — Polars / Arrow integration

**Why.** Modern data science expects `db.cypher(...).to_polars()` or
`.to_arrow()`. Kuzu had this; users coming from there expect it.
Zero-copy export reduces friction for analytical workflows.

**Scope.** Add `.to_polars()` and `.to_arrow()` methods to
`ResultView` next to the existing `.to_list()` / `.to_dict()` /
`.to_pandas()`. Use Arrow IPC for the underlying transport.

**Out of scope.** Polars-as-storage (we're a graph DB, not a DataFrame
engine).

**Effort.** 1-2 weeks.

**Depends on.** Nothing.

---

### §7 — JVM binding

**Why.** Reaches Java, Kotlin, Scala, Clojure. Bolt covers most of this
(JVM has excellent Neo4j clients), but native embedded JVM gives a
faster path for inside-process use.

**Scope.** JNI binding via `jni-rs`. Same API shape as Python.

**Out of scope.** Scala-idiomatic surface; keep parity with Python.

**Effort.** 2-3 weeks.

**Depends on.** §3 (C ABI gives the easier path here).

---

### §8 — Go binding + DAWGS driver

**Why.** Reaches BloodHound + the cloud-native ecosystem (Tailscale-style
infra tools, Kubernetes operators). DAWGS driver makes us a drop-in
storage backend for SpecterOps's stack specifically.

**Scope.** Cgo binding via the §3 C ABI. Once that exists, a DAWGS
driver is a focused 2-3 week project. Lives in
`crates/kglite-dawgs` (Go module, separate go.mod, vendored Rust .so).

**Out of scope.** Making KGLite a drop-in for every Go graph project —
DAWGS-specific is the right scope until pull from elsewhere appears.

**Effort.** 1-2 weeks for binding, 2-3 weeks for the DAWGS driver.

**Depends on.** §3 (C ABI) and §1 (Bolt is the alternative path for
BloodHound and most consumers).

---

### §9 — GQL (ISO 39075) coverage

**Why.** GQL is the ISO standard graph query language as of 2024.
Cypher-derived but with intentional incompatibilities. Eventually the
right thing to track.

**Scope.** When the GQL test suite stabilises and there's a real user
pull, add a GQL parser front-end that lowers to the same AST as our
Cypher parser. Most differences from Cypher are syntactic.

**Out of scope.** Today. This is "watch and wait" until adoption picks up.

**Effort.** Unknown. 4-6 weeks once we commit.

**Depends on.** Real user demand.

---

## Sequencing summary

| Quarter | Focus | Headline ship |
|---|---|---|
| 2026 Q3 (now) | §1 Bolt protocol 🚧 | `kglite-bolt-server` — drop-in for Neo4j-attached tools (see [`bolt_implementation.md`](bolt_implementation.md)) |
| 2026 Q4 | §2 Node.js binding, §3 C ABI, §4 connection-string convention | "We're embeddable from every major language." |
| 2027 Q1 | §5 multi-label, §6 Polars/Arrow | "We pass openCypher conformance more completely, and modern data tools speak to us natively." |
| 2027 Q2+ | §7 JVM, §8 Go + DAWGS, ongoing Cypher coverage | "We're the embedded openCypher engine, period." |

---

## How to read this roadmap

- These are **directional** items, not commitments. Each gets a real
  scoping doc + plan-mode session before work starts.
- Order is leverage-driven, not chronological — §3 (C ABI) might land
  before §2 (Node) if a real Node user appears with a need first, etc.
- "Effort" estimates assume a focused engineer who knows the codebase;
  add buffer for context-switching, integration testing, and the
  inevitable platform-divergence surprises (q.v. the 0.9.52 CI saga).
- When something here ships, move its entry to `CHANGELOG.md` and
  delete from this roadmap. The roadmap is forward-looking only.
