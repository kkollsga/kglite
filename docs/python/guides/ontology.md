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
            "inverse_name": "HAS_STUDENT",      # reading-direction alias
            "inverse_enforced": True,            # opt-in: audit physical pairing
            "cardinality": {"min": 1},
            "required": True,
            "required_properties": ["since"],    # audited per edge
            "property_types": {"since": "integer"},
            "enforcement": "warn",               # or per-check: {"required_properties": "error"}
            "description": "Active enrolment",
        },
        "STRAT_PARENT": {"domain": "Stratigraphy", "range": "Stratigraphy",
                          "ancestry": True},   # parent pointers, walked with *1..
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
- There is an enforced class cap of **512 classes**
  (`MAX_ONTOLOGY_CLASSES`) — a hard number, so capacity planning needs no
  experiment: the layer is for schema-level vocabularies. A million-class
  taxonomy (Wikidata's P279) is **data** — keep it as edges, declare the
  relationship `ancestry: True`, and walk it with `*1..` paths. That
  boundary is a feature. Do **not** reach for `transitive: True` there —
  see the next bullet.
- `transitive: True` and `ancestry: True` both say "this edge is a
  hierarchy", and they are mutually exclusive (declaring both is refused).
  `transitive` is a **promise that the closure is stored**: it enrolls
  `transitivity_violation`, which flags every `a→b→c` with no stored `a→c`
  edge, so a taxonomy that stores only parent pointers reports 100%
  violations. `ancestry` is the annotation for that shape: it records that
  the chain is meaningful and is walked with `*1..`, shows up in
  `describe()`, and enrolls no check.
- `cardinality` / `required` describe **outgoing** edges of the domain type.
- `symmetric: True` lowers to an inverse check of the relationship against
  itself.
- `required_properties` and `property_types` are audited **per edge** of the
  relationship: a listed property must be present and non-null; a declared
  type is checked only on present values. Type names are validated at
  declaration time (the matcher is permissive, so a typo would otherwise
  never fail anything). Both name the **stored** property, so when a
  blueprint renames a loaded column (`rename` on a junction edge — see
  {doc}`blueprints`) declare the name it lands under, not the CSV column.
- `inverse_name` is a **reading-direction alias** — no second edge exists or
  is implied, and it enrolls no check. `inverse_enforced: True` opts into
  auditing physical pairing (each edge must have a stored inverse partner);
  `symmetric: True` keeps its self-inverse check regardless, because
  symmetry *is* a physical claim.
- `enforcement` (`advisory` — the default — / `warn` / `error`) is data for
  the consumers below, never an engine write-guarantee. It also accepts a
  per-check map — `{"required_properties": "error", "domain": "warn"}` —
  where unlisted checks keep the advisory base; keys are the check names
  the audit's `rule` column uses (`domain`, `range`, `required`,
  `required_properties`, `property_types`, `cardinality`, `inverse`,
  `symmetric`, `transitive`).
- `exempt` names, per check, source classes whose violations are counted
  separately instead of against severity — see [Exempting an upstream
  source](#exempting-an-upstream-source).

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

The scorecard rolls every declared check up into one call — one row per
declared check, carrying its violation count, denominator, percentage,
declared severity, and the count its `exempt` classes excused:

```python
from kglite import KnowledgeGraph

g = KnowledgeGraph()
g.cypher("""
CREATE (s:Student {id: 's1'}), (t:Teacher {id: 't1'}), (a:Alumnus {id: 'a1'}),
       (c:Class {id: 'c1'})
CREATE (s)-[:ENROLLED_IN {since: 2024}]->(c)
CREATE (t)-[:ENROLLED_IN {since: 2023}]->(c)
CREATE (a)-[:ENROLLED_IN {since: 2019}]->(c)
""")
g.define_ontology({
    "classes": {"Person": {"abstract": True}, "Student": {"is_a": "Person"},
                "Teacher": {"is_a": "Person"}, "Alumnus": {"is_a": "Person"},
                "Class": {}},
    "relationships": {
        "ENROLLED_IN": {"domain": "Student", "range": "Class",
                        "enforcement": "warn"},
    },
})

for row in g.cypher(
    "CALL ontology_audit() YIELD rule, severity, violations, exempted, total, pct"
):
    print(row)
# {'rule': 'ENROLLED_IN.domain', 'severity': 'warn', 'violations': 2, 'exempted': 0, 'total': 3, 'pct': 66.7}
# {'rule': 'ENROLLED_IN.range', 'severity': 'warn', 'violations': 0, 'exempted': 0, 'total': 3, 'pct': 0.0}
```

Run it after every rebuild and you have data-quality-over-time for free;
an agent can call it cold and qualify its own answers.

**Which source types are violating?** That is the next question every
scorecard raises, and `{by: 'domain_class'}` answers it: each rule's row
fans out into one row per primary node type its violations come from, with
that class's share of `violations` and `pct` (they sum back to the rule's
aggregate) while `severity`, `exempted` and `total` keep their per-rule
values. Exempted rows are left out, so a class whose every violation is
excused gets no row at all, and a rule with nothing to break down keeps its
single aggregate row. Without the parameter, `domain_class` is `None` on
every row — a bare `CALL ontology_audit()` returns all seven columns either
way.

```python
for row in g.cypher(
    "CALL ontology_audit({by: 'domain_class'}) YIELD rule, domain_class, violations, pct"
):
    print(row)
# {'rule': 'ENROLLED_IN.domain', 'domain_class': 'Alumnus', 'violations': 1, 'pct': 33.3}
# {'rule': 'ENROLLED_IN.domain', 'domain_class': 'Teacher', 'violations': 1, 'pct': 33.3}
# {'rule': 'ENROLLED_IN.range', 'domain_class': None, 'violations': 0, 'pct': 0.0}
```

The domain-side class is the edge's source for `domain` / `range` /
`required_properties` / `property_types`, the node itself for `required` /
`cardinality`, and for the pair and triple shapes (`inverse`, `symmetric`,
`transitive`) the first bound node.

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

## Exempting an upstream source

An abstract `domain` is what lets one declaration cover a union edge —
`HAS_OPERATOR` from every `Licensable` — and it is also what makes a single
nonconforming source poison the whole rule. If one upstream register never
carried the date the others do, `required_properties: ["validFrom"]` can
never be promoted past `advisory`: the rule you want to enforce for the
sources you control is permanently red because of a source you do not.

`exempt` is the seam. It is a **per-check map** of source classes whose
violations are counted separately instead of against severity:

```python
from kglite import KnowledgeGraph

g = KnowledgeGraph()
g.cypher("""
CREATE (a:Licence {id: 'PL001'}), (b:Licence {id: 'PL002'}),
       (p:PetregLicence {id: 'P900'}), (c:Company {id: 'EQNR'})
CREATE (a)-[:HAS_OPERATOR {validFrom: 1995}]->(c)
CREATE (b)-[:HAS_OPERATOR]->(c)
CREATE (p)-[:HAS_OPERATOR]->(c)
""")
g.define_ontology({
    "classes": {"Licensable": {"abstract": True},
                "Licence": {"is_a": "Licensable"},
                "PetregLicence": {"is_a": "Licensable"},
                "Company": {}},
    "relationships": {
        "HAS_OPERATOR": {
            "domain": "Licensable", "range": "Company",
            "required_properties": ["validFrom"],
            "enforcement": {"required_properties": "error"},
            # the petroleum register has no start date; the others must have one
            "exempt": {"required_properties": ["PetregLicence"]},
        },
    },
})

for row in g.cypher(
    "CALL ontology_audit() YIELD rule, severity, violations, exempted, total"
):
    print(row)
# {'rule': 'HAS_OPERATOR.domain', 'severity': 'advisory', 'violations': 0, 'exempted': 0, 'total': 3}
# {'rule': 'HAS_OPERATOR.range', 'severity': 'advisory', 'violations': 0, 'exempted': 0, 'total': 3}
# {'rule': 'HAS_OPERATOR.required_properties', 'severity': 'error', 'violations': 1, 'exempted': 1, 'total': 3}
```

Both edges lack `validFrom`; only the `Licence` one counts as a violation,
so the rule can sit at `error` and still block exactly the debt you own.

What the form guarantees:

- **Per-check, never flat.** `exempt: ["PetregLicence"]` is refused — an
  exemption spread silently across every check is not something you can
  reason about later. Name the check it applies to.
- **`required_properties` and `property_types` only.** These are the two
  checks where "the class to exempt" unambiguously means the edge's *source*
  type. Any other check name under `exempt` is refused at declaration time
  with the reason, not just an accept-list.
- **Ancestor-widening.** A class matches when it is the edge source's
  primary type *or* one of its declared ancestors, the same widening
  `domain`/`range` get — exempting `Licensable` exempts the whole subtree.
- **The class must be declared.** An undeclared name is refused: since
  matching widens over the `is_a` forest, a typo would silently exempt
  nothing, which is the exact failure the feature exists to remove.

`exempted` never hides rows. `violations + exempted` is everything the check
flagged, and `edge_property_violation()` lists those individual edges — the
row-level drill-down behind the `required_properties` / `property_types`
counts, with `exempt` marking which side of the line each row fell on:

```python
for row in g.cypher("""
    CALL edge_property_violation() YIELD check, source, property, exempt
    RETURN check, source.id AS source, property, exempt
"""):
    print(row)
# {'check': 'required_properties', 'source': 'PL002', 'property': 'validFrom', 'exempt': False}
# {'check': 'required_properties', 'source': 'P900', 'property': 'validFrom', 'exempt': True}
```

At the blueprint gate the exempted count is reported, never dropped: every
summary line carries a `(+N exempted)` tail, and a rule declared `error`
whose violations are *all* exempted is reported as a **warning** rather than
passing silently. An exemption that quietly absorbed every flagged row
would make a passing gate indistinguishable from a clean graph.

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

### Where you write the label decides the plan

A materialized supertype is worth having only if your queries reach it
through the label engine, and that depends on *where in the query the label
sits* — not on whether the name is spelled the same:

```python
from kglite import KnowledgeGraph

g = KnowledgeGraph()
g.cypher("""CREATE (:Student {id: 1, title: 'Ann'}), (:Teacher {id: 2, title: 'Bo'}),
                  (:Class {id: 3, title: 'Math'})""")
g.define_ontology({"classes": {"Person": {"abstract": True},
                               "Student": {"is_a": "Person"},
                               "Teacher": {"is_a": "Person"}}})
g.materialize_ontology()
g.create_index("Student", "title")     # every live member must be covered
g.create_index("Teacher", "title")

for row in g.cypher("EXPLAIN MATCH (p:Person {title: 'Ann'}) RETURN p.id"):
    print(row)
# {'step': 1, 'operation': 'Match :Person', 'estimated_rows': 2}
# {'step': 2, 'operation': 'ClosureProbe :Person (Student, Teacher)', 'estimated_rows': None}
# {'step': 3, 'operation': 'Return', 'estimated_rows': None}

for row in g.cypher("EXPLAIN MATCH (p) WHERE p:Person AND p.title = 'Ann' RETURN p.id"):
    print(row)
# {'step': 1, 'operation': 'Match', 'estimated_rows': 3}
# {'step': 2, 'operation': 'Where', 'estimated_rows': None}
# {'step': 3, 'operation': 'Return', 'estimated_rows': None}
# {'step': 4, 'operation': 'OptimizerPass push_where_into_match.1', 'estimated_rows': None}
```

- **Pattern position — `MATCH (p:Person)`** is the label engine: the label
  *is* the candidate set. On a `closed` label whose every live member type
  carries an index for the filtered property, a property-filtered supertype
  match runs per-member index probes instead of scanning, and `EXPLAIN` says
  so with a `ClosureProbe :Person (Student, Teacher)` row naming the members
  it would visit. No row means no probe — the label is `open`, a member is
  unindexed, or the label is not materialized at all, and the match falls
  back to a scan that is still correct. (A value written as a parameter,
  `{title: $t}`, is unresolved when the plan renders, so the marker stays
  off; the runtime probe still applies.)
- **`WHERE p:Person`** is an ordinary post-candidate predicate. The pattern
  binds every node in the graph and the label is checked per row — note the
  unlabelled `Match` above, estimating all 3 nodes rather than the 2 that
  carry `:Person`. Nothing rewrites a `WHERE`-position label check into a
  candidate set, so this shape never probes. Move the label into the
  pattern.
- **Alternation — `MATCH (p:Student|Teacher)`** is the *unmaterialized*
  alternative: it matches the union of the branches with no labels stamped
  and no closure to maintain. It carries no `ClosureProbe` either — there is
  no managed bucket to trust — and it names the members literally, so a
  subtype added to the class forest later will not be in it. Prefer it when
  you want the union once; materialize when the supertype is a first-class
  thing your queries name repeatedly.

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

## How this maps to RDFS / OWL / SHACL

For readers coming from the semantic-web stack, the honest positioning —
including three places where a familiar word carries *different* semantics
here:

| Concept | There | Here |
|---|---|---|
| `is_a` | `rdfs:subClassOf`, a DAG (multiple inheritance) | a **forest** — one parent per class. Multi-role is modelled on *nodes* (secondary labels; a materialized node carries the union of its labels' ancestries), not in the class graph. |
| `domain` / `range` | RDFS **infers**: using an edge *entails* the subject's class membership | **checked, never inferred** — the SHACL reading. A violating edge is reported (or refused at the build gate); nothing ever gains a class from edge use. If you expect RDFS semantics, this is the one difference to internalize. |
| Validation | SHACL, deliberately separate from ontology/inference | the same separation, built in: `enforcement: advisory \| warn \| error` maps onto `sh:Info` / `sh:Warning` / `sh:Violation`. |
| Subclass queries | entailment regimes; production stores typically **materialize** entailments | the same technique: `materialize_ontology()` stamps ancestor labels; no query rewriting, no reasoner. |
| `inverse_name` | `owl:inverseOf` creates/entails the inverse triples | naming only — Cypher already traverses both directions, so no second edge exists or is implied. `inverse_enforced: True` opts into auditing stored pairing instead. |
| `transitive` | `owl:TransitiveProperty`, entailed closures | nothing is entailed: it declares that the closure is **stored**, and `transitivity_violation` audits that claim (every `a→b→c` needs a stored `a→c`). |
| `ancestry` | no counterpart — a reasoner would entail the chain | documentation only: the chain is meaningful and is *walked* (`*1..`), never stored. This is what a parent-pointer taxonomy declares. |
| "abstract" | not an ontology notion (any class may have instances) | borrowed from the schema world: a class that names no node type and cannot be instantiated directly. |

**Deliberate non-goals** (not omissions): entailment of any kind,
open-world semantics, equivalence classes (`owl:equivalentClass` — within
one graph, two names for one concept is a rebuild, not an axiom),
restriction classes, property hierarchies (`rdfs:subPropertyOf`), and
disjointness axioms (low value under single primary types). RDFS/SKOS
import/export is tracked as future interop, adopting the vocabulary, not
the entailment.

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
