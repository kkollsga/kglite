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
| **Wire protocol** | Embedded only | **Bolt v5 server — shipped (0.10.x)** | **Done.** `kglite-bolt-server` speaks the Neo4j Bolt wire protocol; the whole Neo4j ecosystem plugs in unchanged. |

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

### §1 — Node.js binding

**Why.** Node/TS is where the LLM-tooling ecosystem lives (LangChain.js,
llama-index TS, Mastra, every browser-attached agent). A Python-only
graph DB cuts ourselves off from half the agent market.

**Scope.** napi-rs binding exposing the same surface as the PyO3
binding: `KnowledgeGraph`, `cypher()`, `add_nodes()`, fluent API,
`describe()`. Ships as `@kglite/kglite` npm package. Reuses the existing
Rust core unchanged.

**Out of scope.** A separate JS API surface — match Python's shape.

**Effort.** 1-2 weeks; independent of the other items.

**Depends on.** A stable Rust API at `kglite::api::*`. Phase G
(2026-05-24) shipped this; the pyo3-free pure-Rust core is the
crate that future bindings wrap.

---

### §2 — Connection-string convention

**Why.** Every modern data tool expects `<scheme>://<path>` configuration.
Bolt already gives us `bolt://localhost:7687`. For embedded use we
should also have `kglite://./graphs/mygraph.kgl` consistently — that's
how tools wire up DSNs.

**Scope.** Standardise `kglite://` in the README, docs, and any
constructor that takes a path. Add a `KnowledgeGraph.from_url()`
constructor that accepts both `kglite://` (embedded) and `bolt://`
(client mode — talks to the shipped Bolt server).

**Out of scope.** Parsing every variant. Just the two.

**Effort.** Days.

**Depends on.** the Bolt server (shipped) for the `bolt://` half.

---

### §3 — Polars / Arrow integration

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

### §4 — JVM binding

**Why.** Reaches Java, Kotlin, Scala, Clojure. Bolt covers most of this
(JVM has excellent Neo4j clients), but native embedded JVM gives a
faster path for inside-process use.

**Scope.** JNI binding via `jni-rs`. Same API shape as Python.

**Out of scope.** Scala-idiomatic surface; keep parity with Python.

**Effort.** 2-3 weeks.

**Depends on.** the C ABI (shipped 0.10.3) — gives the easier path here.

---

### §5 — Go binding + DAWGS driver

**Why.** Reaches BloodHound + the cloud-native ecosystem (Tailscale-style
infra tools, Kubernetes operators). DAWGS driver makes us a drop-in
storage backend for SpecterOps's stack specifically.

**Scope.** Cgo binding via the C ABI (shipped 0.10.3 — `crates/kglite-c`).
Sibling repo `kkollsga/kglite-go`, separate go.mod, pre-built
`libkglite_c.{so,dylib,dll}` artifacts via GitHub releases. Tiered
ambition: minimum-viable (~600 LOC, 2-3 days) → production-ready
(~1500-2000 LOC, 1-2 weeks) → "proper" idiomatic library (3-4 weeks).
A DAWGS driver layers on top of the Go binding.

**Out of scope.** Making KGLite a drop-in for every Go graph project —
DAWGS-specific is the right scope until pull from elsewhere appears.

**Effort.** 1-2 weeks for production-ready binding, 2-3 weeks for the
DAWGS driver.

**Depends on.** the C ABI (shipped 0.10.3). The Bolt server (shipped) is
the alternative path for BloodHound and most consumers.

---

### §6 — GQL (ISO 39075) coverage

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
| _shipped (0.10.x)_ | Bolt protocol server | `kglite-bolt-server` — drop-in for every Neo4j-attached tool |
| 2026 Q3 (now) | §1 Node.js binding, §2 connection-string convention | "We're embeddable from every major language." |
| 2027 Q1 | §3 Polars/Arrow | "Modern data tools speak to us natively." |
| 2027 Q2+ | §4 JVM, §5 Go + DAWGS, §6 GQL, ongoing Cypher coverage | "We're the embedded openCypher engine, period." |

---

## How to read this roadmap

- These are **directional** items, not commitments. Each gets a real
  scoping doc + plan-mode session before work starts.
- Order is leverage-driven, not chronological — a lower-numbered item
  might land after a higher-numbered one if a real user appears with a
  need first (this is how the C ABI, now shipped, landed ahead of the
  Node binding).
- "Effort" estimates assume a focused engineer who knows the codebase;
  add buffer for context-switching, integration testing, and the
  inevitable platform-divergence surprises (q.v. the 0.9.52 CI saga).
- When something here ships, move its entry to `CHANGELOG.md` and
  delete from this roadmap. The roadmap is forward-looking only.
