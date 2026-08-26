# Ontology (declared semantic layer)

An ontology gives type names a declared "kind of" structure — `Student`
*is a* `Person`, `Licence` *is a* `Licensable` — plus machine-readable
semantics for relationships: which types an edge connects, which properties
it must carry, its inverse's name, its cardinality. KGLite persists these
declarations with the graph and wires them into `describe()`, the rule
procedures, blueprint builds, and (optionally) label matching itself.

**The scope fence: annotations, not axioms.** In spirit this is SKOS, not
OWL. The ontology never invents facts and never changes what a query
matches on its own — no entailment, no open-world semantics, no reasoning.
Its three jobs are: state the concept model machines can read, provide
defaults for validators you already have, and (opt-in) act as a
data-quality contract at build time.

It is also deliberately **not** `set_parent_type`: that map is presentation
*ownership* (which types are supporting detail in `describe()` tiering); the
ontology is semantic *kind-of*. `WellboreCore → Wellbore` is ownership;
`Licence is_a Licensable` is an ontology fact. Neither is derived from the
other.

## Declaring

```python
g.define_ontology({
    "classes": {
        "Person":  {"abstract": True, "description": "Any human actor"},
        "Student": {"is_a": "Person"},
        "Teacher": {"is_a": "Person"},
        # documentation-only discriminator (rendered as unenforced):
        "Wellbore": {"by": "wlbWellType"},
    },
    "relationships": {
        "ENROLLED_IN": {
            "domain": "Student", "range": "Class",
            "inverse_name": "HAS_STUDENT",
            "cardinality": {"min": 1},
            "required": True,
            "required_properties": ["since"],
            "enforcement": "warn",
            "description": "Active enrolment",
        },
        "STRAT_PARENT": {"domain": "Stratigraphy", "range": "Stratigraphy",
                          "transitive": True},
    },
})
```

Rules the declaration must satisfy (checked on install):

- `is_a` is a **forest**: one parent per class, no cycles, parents must be
  declared. Multi-role nodes are modelled with secondary labels on nodes,
  not multiple inheritance in the class graph.
- Class names share the label namespace. A class naming a live node type is
  *concrete*; a class naming none must usually be `abstract: True`
  (concrete-with-no-instances is a returned warning, abstract-shadowing-a-
  live-type is an error) — `MATCH (n:X)` keeps exactly one meaning per name.
- There is an enforced class cap (hundreds): the layer is for schema-level
  vocabularies. A million-class taxonomy (Wikidata's P279) is **data** —
  keep it as edges, declare the relationship `transitive: True`, and walk it
  with `*1..` paths. That boundary is a feature.
- `cardinality` / `required` describe **outgoing** edges of the domain type.
- `symmetric: True` lowers to an inverse check of the relationship against
  itself.
- `enforcement` (`advisory` — the default — / `warn` / `error`) is data for
  the consumers below, never an engine write-guarantee.

`g.ontology()` returns the store as a dict, `g.clear_ontology()` removes it
(withdrawing any materialized labels first). The store persists in the
`.kgl` and travels with `save_subset` / `to_subgraph`.

## Reading it back

- `SHOW ONTOLOGY` — one row per class and relationship.
- `describe()` — an `<ontology>` section renders whenever a store is
  declared (focused mode narrows to the classes touching the requested
  types). No new mode or parameter: absence of the section *is* the "no
  ontology" signal.
- `describe(cypher=["ontology"])` — the agent-facing topic documentation.

## Declaration-driven validators

The six declaration-backed rule procedures called with **no arguments**
check every relevant declaration, each row carrying a `rule` column naming
the declaration it came from:

```cypher
CALL type_domain_violation() YIELD source, target, rule
CALL missing_required_edge() YIELD node, rule
CALL inverse_violation() YIELD a, b, rule
```

A `domain`/`range` naming an **abstract class widens to its declared
descendants** — this is the union-endpoint case a flat schema cannot
declare (`HAS_OPERATOR` from six concrete source types becomes
`domain: "Licensable"`, and the existing checks finally reach it).

The scorecard rolls every declared check up into one call:

```cypher
CALL ontology_audit() YIELD rule, severity, violations, total, pct
-- ENROLLED_IN.domain     warn   1/2    50.0
-- ENROLLED_IN.required   warn   1/2    50.0
```

Run it after every rebuild and you have data-quality-over-time for free;
an agent can call it cold and qualify its own answers.

## The blueprint gate (observe → fix → enforce)

Reference the document from a blueprint and the declarations become a
build-time contract:

```json
{ "ontology": "school.ontology.json", "nodes": { ... } }
```

The gate runs as a final build phase, after all loading, before anything is
saved. Per-declaration severity decides what a violation does:

| `enforcement` | On violation |
|---|---|
| `advisory` | nothing at build time — available on demand via the `CALL`s |
| `warn` | one summary line per rule in the build report |
| `error` | **report every violation, then fail once** — no output file is written |

The intended lifecycle: start every rule at `advisory`, read the report as
your cleanup worklist, fix the data, then flip the rules you own to
`error` so the debt can never silently return. Rules describing *upstream*
data reality stay `warn` forever — they belong in the build log, not the
exit code.

## Materialization (making supertypes matchable)

Everything above changes no query semantics. Materialization does — by
explicit opt-in, and through completely ordinary machinery:

```python
g.materialize_ontology()
g.cypher("MATCH (p:Person) RETURN p.name")   # finds Students and Teachers
```

`Student is_a Person` is stamped as the **real secondary label** `:Person`
on every `Student` node, through the same bulk label path every label write
uses — so `MATCH (p:Person)` works with today's semantics, today's
candidate index, today's `EXPLAIN`, and `labels(n)`, CDC, exports, and Bolt
clients never disagree with what queries see. From then on the write paths
maintain the closure: a created `Student` carries `:Person` from birth, and
creating a node of a declared *abstract* class is refused, naming the
concrete subtypes.

Each materialized label is **managed**, in one of two states
(`g.ontology_diff()` reports them):

- **`closed`** — the engine is the bucket's only writer; the label holds
  exactly the declared closure. Closure-reliant optimizations may trust it:
  a property-filtered supertype match (`MATCH (p:Person {name: 'Ann'})`)
  runs per-descendant index probes instead of scanning.
- **`open`** — something outside the closure touched the label (a manual
  `SET n:Person` on a non-member, an adopted pre-existing bucket, an
  extend-graph union). Everything stays *correct*; the optimizations switch
  off for that label.

Writers downgrade to `open` rather than refuse — a performance cliff
instead of a wrong-answer cliff. The one refusal is manual `REMOVE` of a
managed label (an under-complete bucket has no safe state);
`g.dematerialize_ontology()` is the exit, and it recovers correctly through
the write-ahead log like every other write.

Materializing onto a graph whose label buckets already have members the
closure cannot explain is refused unless `materialize_ontology(adopt=True)`
(the label is then managed `open`).

## Serving it over MCP

```yaml
extensions:
  ontology:
    file: school.ontology.json
    materialize: true        # optional
```

The server installs (and optionally materializes) the declarations at boot,
**memory-only**: nothing in the server auto-saves, so the source `.kgl` is
untouched — adoption with zero build-script changes. An agent explicitly
calling the `save_graph` tool persists them, which is then correct.

## When not to use it

- **Large taxonomies as classes** — enforced away by the class cap; keep
  them as edges (see the declaration rules above).
- **Entity resolution** — the ontology relates *types*, never records.
  `Student is_a Person` says nothing about whether two Alice Smiths are one
  human.
- **As a data-cleaning tool** — the gate *finds and then guards* cleanup;
  the fixing itself belongs in your load pipeline.

## See also

- {doc}`blueprints` — the build pipeline the gate plugs into.
- {doc}`traversal-hierarchy` — `set_parent_type`, the *presentation*
  hierarchy this layer is deliberately not.
- {doc}`/concepts/multi-label-rationale` — the label model the
  materialization builds on.
