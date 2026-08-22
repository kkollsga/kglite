//! Non-fatal query warnings — the "why did this return nothing?" family.
//!
//! Split out of [`super`] when the combined file passed the god-file ceiling.
//! The fatal schema check stays there; everything here *only ever appends to a
//! `Vec<String>`*, and that separation is the point: a bug in this file can
//! make a message wrong or missing, never a valid query rejected.
//!
//! The families and their conservatism rules are documented on the module
//! doc-comment of [`super`] and on the individual functions.

use super::super::super::ast::*;
use super::{for_each_query_pattern, PatternSite, SchemaError, SchemaErrorKind, BUILTIN_FIELDS};
use crate::graph::core::pattern_matching::{EdgeDirection, NodePattern, Pattern, PatternElement};
use crate::graph::mutation::validation::did_you_mean;
use crate::graph::schema::{DirGraph, InternedKey};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU8, Ordering};

/// var → single known node label, from MATCH / OPTIONAL MATCH node patterns.
///
/// Multi-label and unknown-label vars are dropped (can't reason precisely),
/// and a var introduced by a *projection* (`WITH n.x AS y`, `WITH n AS m`) is
/// never entered: only a MATCH pattern binds a name to a label this map can
/// trust, so an aliased var is simply absent and every check that consults the
/// map stays silent about it. Both warning families below share the map, so
/// they share that boundary too.
fn match_var_labels<'q>(query: &'q CypherQuery, graph: &DirGraph) -> HashMap<&'q str, &'q str> {
    let mut var_label: HashMap<&str, &str> = HashMap::new();
    if graph.node_type_metadata.is_empty() {
        return var_label;
    }
    for clause in &query.clauses {
        if let Clause::Match(m) | Clause::OptionalMatch(m) = clause {
            for pattern in &m.patterns {
                for el in &pattern.elements {
                    if let PatternElement::Node(np) = el {
                        if let (Some(var), Some(label)) =
                            (np.variable.as_deref(), np.node_type.as_deref())
                        {
                            if np.extra_labels.is_empty()
                                && graph.node_type_metadata.contains_key(label)
                            {
                                var_label.insert(var, label);
                            } else {
                                var_label.remove(var);
                            }
                        }
                    }
                }
            }
        }
    }
    var_label
}

/// Where an absent-property reference was found. The wording differs per site
/// because the *failure* differs: a filter silently drops every row, a
/// projection silently fills a column with nulls (the eval's `RETURN v.imo`
/// case — deadlier because a sibling `v.name` title-aliases to a real value,
/// so the rows read as half-correct), and a sort key that is always null
/// silently does nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbsentSite {
    Where,
    Return,
    With,
    OrderBy,
}

impl AbsentSite {
    /// The clause keyword, for the locked-schema *error* voice — which states
    /// where the mistake is rather than what the query would have silently
    /// done, because under a lock the query does not run at all.
    fn clause(self) -> &'static str {
        match self {
            AbsentSite::Where => "WHERE",
            AbsentSite::Return => "RETURN",
            AbsentSite::With => "WITH",
            AbsentSite::OrderBy => "ORDER BY",
        }
    }

    fn message(self, property: &str, label: &str, hint: &str) -> String {
        match self {
            AbsentSite::Where => format!(
                "WHERE references property '{property}' which no {label} node has — the \
                 comparison is null (always false), so this filters out every row.{hint}"
            ),
            AbsentSite::Return => format!(
                "RETURN projects property '{property}' which no {label} node has — every \
                 value will be null.{hint}"
            ),
            AbsentSite::With => format!(
                "WITH projects property '{property}' which no {label} node has — every \
                 value will be null.{hint}"
            ),
            AbsentSite::OrderBy => format!(
                "ORDER BY sorts on property '{property}' which no {label} node has — every \
                 key is null, so the sort does nothing.{hint}"
            ),
        }
    }
}

/// One absent-property finding, kept structured rather than pre-rendered.
///
/// The *same* finding reads two ways depending on the schema state, and the
/// disposition is not known where it is discovered: on an open schema it is a
/// warning about what the query will silently do, and under
/// [`DirGraph::schema_locked`] it is the error that stops the query
/// ([`strict_read_error`]). Keeping the parts lets one walk serve both instead
/// of the caller string-matching prose to tell the families apart.
#[derive(Debug, Clone)]
pub(crate) struct AbsentProperty {
    site: AbsentSite,
    property: String,
    label: String,
    /// Pre-rendered `" Did you mean 'x'?"` (empty when nothing is close).
    hint: String,
}

impl AbsentProperty {
    fn warning(&self) -> String {
        self.site.message(&self.property, &self.label, &self.hint)
    }
}

/// Best-effort findings for `var.prop` where `prop` exists on **no** node of
/// `var`'s label — in a `WHERE`, where `null <op> x` is false and the
/// predicate silently filters out every row (operator feedback A1b
/// 2026-06-17), and in a `RETURN` / `WITH` / `ORDER BY`, where the column is
/// silently all-null (external eval 2026-08-20 §3a).
///
/// Non-fatal by default: a legitimately-sparse property is still in the
/// type's metadata (set on ≥1 node), so only a *genuinely-absent* property
/// trips this — no false positive on nullable columns. A property the same
/// query writes is likewise not absent by the time it is read back; see
/// [`AbsentPropertyScan::written`]. `lock_schema()` opts into rejecting them
/// instead — see [`strict_read_error`], which reuses these very findings.
fn absent_property_findings<'q>(
    query: &'q CypherQuery,
    graph: &DirGraph,
    var_label: &HashMap<&'q str, &'q str>,
) -> Vec<AbsentProperty> {
    if var_label.is_empty() {
        return Vec::new();
    }
    let mut scan = AbsentPropertyScan {
        graph,
        var_label,
        written: HashSet::new(),
        written_any: HashSet::new(),
        seen: HashSet::new(),
        out: Vec::new(),
    };
    collect_written_properties(&query.clauses, &mut scan.written, &mut scan.written_any);
    for clause in &query.clauses {
        match clause {
            Clause::Where(w) => scan.predicate(&w.predicate, AbsentSite::Where),
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                if let Some(wc) = &m.where_clause {
                    scan.predicate(&wc.predicate, AbsentSite::Where);
                }
            }
            Clause::With(w) => {
                for item in &w.items {
                    scan.expression(&item.expression, AbsentSite::With);
                }
                if let Some(wc) = &w.where_clause {
                    scan.predicate(&wc.predicate, AbsentSite::Where);
                }
            }
            Clause::Return(r) => {
                for item in &r.items {
                    scan.expression(&item.expression, AbsentSite::Return);
                }
            }
            Clause::OrderBy(o) => {
                for item in &o.items {
                    scan.expression(&item.expression, AbsentSite::OrderBy);
                }
            }
            _ => {}
        }
    }
    scan.out
}

/// True when `prop` is neither a built-in field nor in `node_type`'s declared
/// metadata (and the type *has* declared metadata — empty ⇒ skip, as
/// [`validate_property`] does, to avoid false positives on under-declared graphs).
pub(super) fn property_absent(graph: &DirGraph, node_type: &str, prop: &str) -> bool {
    if BUILTIN_FIELDS.contains(&prop) {
        return false;
    }
    match graph.node_type_metadata.get(node_type) {
        Some(tp) => !tp.is_empty() && !tp.contains_key(prop),
        None => false,
    }
}

/// Properties this query writes before anything reads them back.
///
/// The warnings are computed *before* execution, so `SET p.tag = 1 RETURN
/// p.tag` would otherwise be reported as an all-null column — a lie about a
/// property the statement is in the act of creating. `SET n = map` / `SET n +=
/// map` with a non-literal map contributes the variable to `written_any`
/// (nothing narrower is knowable at plan time).
fn collect_written_properties<'q>(
    clauses: &'q [Clause],
    named: &mut HashSet<(&'q str, &'q str)>,
    any: &mut HashSet<&'q str>,
) {
    for clause in clauses {
        match clause {
            Clause::Set(s) => note_written_set_items(&s.items, named, any),
            Clause::Merge(m) => {
                for items in [m.on_create.as_ref(), m.on_match.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    note_written_set_items(items, named, any);
                }
            }
            Clause::Foreach { body, .. } => collect_written_properties(body, named, any),
            Clause::CallSubquery { body, .. } => {
                collect_written_properties(&body.clauses, named, any)
            }
            Clause::Union(u) => collect_written_properties(&u.query.clauses, named, any),
            _ => {}
        }
    }
}

fn note_written_set_items<'q>(
    items: &'q [SetItem],
    named: &mut HashSet<(&'q str, &'q str)>,
    any: &mut HashSet<&'q str>,
) {
    for item in items {
        match item {
            SetItem::Property {
                variable, property, ..
            } => {
                named.insert((variable.as_str(), property.as_str()));
            }
            SetItem::Map {
                variable,
                expression,
                ..
            } => match expression {
                Expression::MapLiteral(entries) => {
                    for (key, _) in entries {
                        named.insert((variable.as_str(), key.as_str()));
                    }
                }
                _ => {
                    any.insert(variable.as_str());
                }
            },
            SetItem::Label { .. } => {}
        }
    }
}

/// One query's absent-property walk: the shared state the predicate and
/// expression recursions thread through, so adding a site is one match arm
/// rather than another parameter on six signatures.
struct AbsentPropertyScan<'a, 'q> {
    graph: &'a DirGraph,
    var_label: &'a HashMap<&'q str, &'q str>,
    /// `(var, prop)` pairs this query writes — see [`collect_written_properties`].
    written: HashSet<(&'q str, &'q str)>,
    /// Vars written wholesale through a non-literal `SET n = map`.
    written_any: HashSet<&'q str>,
    /// `(var, prop)` already reported. First site wins, so the same typo in a
    /// `WHERE` and a `RETURN` is one message, worded for the filter — the more
    /// consequential of the two.
    seen: HashSet<(&'q str, &'q str)>,
    out: Vec<AbsentProperty>,
}

impl<'q> AbsentPropertyScan<'_, 'q> {
    fn report(&mut self, variable: &'q str, property: &'q str, site: AbsentSite) {
        let Some(&label) = self.var_label.get(variable) else {
            return;
        };
        if !property_absent(self.graph, label, property)
            || self.written_any.contains(variable)
            || self.written.contains(&(variable, property))
            || !self.seen.insert((variable, property))
        {
            return;
        }
        let candidates: Vec<&str> = self
            .graph
            .node_type_metadata
            .get(label)
            .map(|m| m.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default();
        self.out.push(AbsentProperty {
            site,
            property: property.to_string(),
            label: label.to_string(),
            hint: did_you_mean(property, &candidates),
        });
    }

    fn predicate(&mut self, pred: &'q Predicate, site: AbsentSite) {
        match pred {
            Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
                self.predicate(a, site);
                self.predicate(b, site);
            }
            Predicate::Not(p) => self.predicate(p, site),
            Predicate::Comparison { left, right, .. } => {
                self.expression(left, site);
                self.expression(right, site);
            }
            Predicate::In { expr, .. }
            | Predicate::InLiteralSet { expr, .. }
            | Predicate::InExpression { expr, .. }
            | Predicate::StartsWith { expr, .. }
            | Predicate::EndsWith { expr, .. }
            | Predicate::Contains { expr, .. }
            | Predicate::IsNull(expr)
            | Predicate::IsNotNull(expr) => self.expression(expr, site),
            _ => {}
        }
    }

    fn expression(&mut self, expr: &'q Expression, site: AbsentSite) {
        match expr {
            Expression::PropertyAccess { variable, property } => {
                self.report(variable.as_str(), property.as_str(), site)
            }
            Expression::Add(a, b)
            | Expression::Subtract(a, b)
            | Expression::Multiply(a, b)
            | Expression::Divide(a, b)
            | Expression::Modulo(a, b)
            | Expression::Concat(a, b) => {
                self.expression(a, site);
                self.expression(b, site);
            }
            Expression::Negate(e) => self.expression(e, site),
            Expression::FunctionCall { args, .. } => {
                for a in args {
                    self.expression(a, site);
                }
            }
            Expression::ListLiteral(items) => {
                for it in items {
                    self.expression(it, site);
                }
            }
            _ => {}
        }
    }
}

/// Non-fatal warning for a relationship pattern whose direction matches no
/// stored edge while the opposite direction has them —
/// `(p:Port)-[:ARRIVES_AT]->(v:Voyage)` when every `ARRIVES_AT` edge runs
/// Voyage→Port. Zero rows, no error, and the single most common LLM Cypher
/// mistake (external eval 2026-08-20 §3b).
///
/// Deliberately conservative: it fires only when the metadata shows the
/// pattern's orientation is unsupported **and** the opposite one is supported.
/// A false positive would be a lie about a pattern that does match, so every
/// ambiguity resolves to silence:
///
/// - [`ConnectionTypeInfo::source_types`] / `target_types` are *unions* over
///   every observed endpoint pair, so a type seen as both A→B and B→A passes
///   the forward test and is never flagged. Multi-pair types can therefore
///   hide a genuinely reversed arrow (accepted false negative); a real A→B
///   edge always guarantees membership, so the check cannot invent one.
/// - Undirected (`-[]-`) patterns have no orientation to be wrong about.
/// - Variable-length expansions are skipped: the endpoint labels of a
///   multi-hop path are not the endpoint types of one stored edge.
/// - An endpoint with no resolvable single label is skipped — `var_label`
///   answers for a bare var another MATCH labelled, and otherwise there is no
///   set to test membership against.
/// - An alternation warns only when *every* branch is reversed, because the
///   pattern still matches through any branch that is not. An unknown branch
///   is not reversed (the unknown-relationship-type warning covers it), so a
///   typo produces one message rather than two.
fn reversed_direction_warnings(
    pattern: &Pattern,
    var_label: &HashMap<&str, &str>,
    graph: &DirGraph,
    seen: &mut HashSet<String>,
    out: &mut Vec<String>,
) {
    for window in pattern.elements.windows(3) {
        let [PatternElement::Node(left), PatternElement::Edge(edge), PatternElement::Node(right)] =
            window
        else {
            continue;
        };
        if edge.var_length.is_some() {
            continue;
        }
        let (from, to) = match edge.direction {
            EdgeDirection::Outgoing => (left, right),
            EdgeDirection::Incoming => (right, left),
            EdgeDirection::Both => continue,
        };
        let (Some(from_label), Some(to_label)) = (
            endpoint_label(from, var_label, graph),
            endpoint_label(to, var_label, graph),
        ) else {
            continue;
        };
        let mut rels: Vec<&str> = Vec::new();
        for rel in edge
            .connection_type
            .iter()
            .chain(edge.connection_types.iter().flatten())
        {
            if !rels.contains(&rel.as_str()) {
                rels.push(rel.as_str());
            }
        }
        if rels.is_empty()
            || !rels
                .iter()
                .all(|rel| is_reversed(graph, rel, from_label, to_label))
        {
            continue;
        }
        let named = rels
            .iter()
            .map(|r| format!("'{r}'"))
            .collect::<Vec<_>>()
            .join(" or ");
        if !seen.insert(format!("D:{named}:{from_label}:{to_label}")) {
            continue;
        }
        let subject = if rels.len() == 1 {
            format!("every {named} relationship")
        } else {
            "every one of them".to_string()
        };
        out.push(format!(
            "MATCH traverses {named} as {from_label} → {to_label}, but {subject} runs \
             {to_label} → {from_label} — this pattern matches no edges. Reverse the arrow?"
        ));
    }
}

/// True when `rel` has no recorded `from → to` endpoint pair but does have a
/// `to → from` one.
fn is_reversed(graph: &DirGraph, rel: &str, from_label: &str, to_label: &str) -> bool {
    let Some(info) = graph.connection_type_metadata.get(rel) else {
        return false;
    };
    let supports =
        |src: &str, tgt: &str| info.source_types.contains(src) && info.target_types.contains(tgt);
    !supports(from_label, to_label) && supports(to_label, from_label)
}

/// The single label an endpoint can be tested against: the pattern's own label
/// when it names one the graph knows, otherwise the label a MATCH bound the
/// bare variable to. `None` for multi-label patterns (no single membership
/// test) and for labels the graph has never seen.
fn endpoint_label<'s>(
    node: &'s NodePattern,
    var_label: &HashMap<&'s str, &'s str>,
    graph: &DirGraph,
) -> Option<&'s str> {
    if !node.extra_labels.is_empty() {
        return None;
    }
    if let Some(label) = node.node_type.as_deref() {
        let known =
            graph.node_type_metadata.contains_key(label) || graph.type_indices.contains_key(label);
        return known.then_some(label);
    }
    node.variable
        .as_deref()
        .and_then(|var| var_label.get(var).copied())
}

/// One statement's non-fatal findings, split by **disposition** rather than by
/// family: [`Self::absent_property`] is the subset `lock_schema()` promotes to
/// a hard error (see [`strict_read_error`]), and [`Self::other`] — unknown
/// labels, unknown relationship types, reversed arrows — stays a warning in
/// every schema state, because each of those shapes is legal Cypher whose
/// zero-row answer is a legitimate thing to ask for.
pub(crate) struct QueryWarnings {
    pub(crate) absent_property: Vec<AbsentProperty>,
    pub(crate) other: Vec<String>,
}

impl QueryWarnings {
    /// Flatten to the wire form every consumer sees — absent-property first,
    /// matching the order the families were emitted in before the split.
    pub(crate) fn into_messages(self) -> Vec<String> {
        let mut out: Vec<String> = self
            .absent_property
            .iter()
            .map(AbsentProperty::warning)
            .collect();
        out.extend(self.other);
        out
    }
}

/// Flat message list — the form `QueryDiagnostics::warnings` and every binding
/// consume. [`collect_query_warnings`] is the same computation with the
/// disposition split still intact, for the one caller (`session::execute`)
/// that must know which findings a locked schema rejects.
pub fn collect_unknown_pattern_warnings(query: &CypherQuery, graph: &DirGraph) -> Vec<String> {
    collect_query_warnings(query, graph).into_messages()
}

/// Non-fatal counterpart to [`validate_schema`]: collect "did you mean?"
/// warnings for MATCH patterns that reference a node label or relationship
/// type the graph has never seen (a zero-row existence check is legal Cypher,
/// so this is *not* an error), plus the absent-property findings from
/// [`absent_property_findings`]. Pure (no I/O), so directly testable;
/// [`emit_query_warnings`] is the stderr side of the same computation.
pub(crate) fn collect_query_warnings(query: &CypherQuery, graph: &DirGraph) -> QueryWarnings {
    let have_node_schema =
        !graph.node_type_metadata.is_empty() || graph.type_indices.keys().next().is_some();
    let have_edge_schema = !graph.connection_type_metadata.is_empty();
    if !have_node_schema && !have_edge_schema {
        return QueryWarnings {
            absent_property: Vec::new(),
            other: Vec::new(),
        };
    }

    // Walk every read pattern — top-level MATCH / OPTIONAL MATCH *and* the
    // ones nested in `CALL {}`, `WHERE EXISTS {}` and `UNION` branches (see
    // [`walk_query_patterns`]) — checking each label/relationship against the
    // schema directly. The all-valid path (the overwhelming common case)
    // allocates nothing: only confirmed-unknown, not-yet-seen names are
    // recorded, and the candidate lists for "did you mean?" are built lazily
    // only if there's at least one unknown.
    let mut seen: HashSet<String> = HashSet::new();
    let mut unknown_labels: Vec<String> = Vec::new();
    // Shared with the absent-property walk below: a bare endpoint var
    // (`MATCH (x:Person) MATCH (a:Paper)-[:AUTHORED]->(x)`) is label-typed
    // through the same map.
    let var_label = match_var_labels(query, graph);
    let mut reversed: Vec<String> = Vec::new();
    // `(unknown type, the branches of its alternation that ARE known)`. The
    // surviving branches decide the wording: a relationship alternation
    // (`-[:A|B]->`) matches through *any* branch, so one unknown branch only
    // means "returns no rows" when every branch is unknown.
    let mut unknown_rels: Vec<(String, Vec<String>)> = Vec::new();

    for_each_query_pattern(query, &mut |site| {
        // Write patterns are deliberately skipped: on an open schema
        // `CREATE (n:NewType)` is how a type comes into existence.
        let PatternSite::Read(pattern) = site else {
            return;
        };
        for element in &pattern.elements {
            match element {
                PatternElement::Node(np) if have_node_schema => {
                    for label in np.node_type.iter().chain(np.extra_labels.iter()) {
                        // A label is known if it's a declared primary
                        // type OR a secondary label applied via
                        // add_label (`MATCH (n:Reviewer)` is valid even
                        // though `Reviewer` is no node's primary type).
                        let known = graph.node_type_metadata.contains_key(label)
                            || graph.type_indices.contains_key(label)
                            || graph
                                .secondary_label_index
                                .contains_key(&InternedKey::from_str(label));
                        if !known && seen.insert(format!("L:{label}")) {
                            unknown_labels.push(label.clone());
                        }
                    }
                }
                PatternElement::Edge(ep) if have_edge_schema => {
                    // Both fields, because a single type lands in
                    // `connection_type` and an alternation in
                    // `connection_types`.
                    let branches = || {
                        ep.connection_type
                            .iter()
                            .chain(ep.connection_types.iter().flatten())
                    };
                    let known = |rel: &String| graph.connection_type_metadata.contains_key(rel);
                    // All-valid stays allocation-free: the surviving-branch
                    // list is only built once an unknown is confirmed.
                    if branches().all(known) {
                        continue;
                    }
                    let surviving: Vec<String> = branches().filter(|r| known(r)).cloned().collect();
                    for rel in branches().filter(|r| !known(r)) {
                        if seen.insert(format!("R:{rel}")) {
                            unknown_rels.push((rel.clone(), surviving.clone()));
                        }
                    }
                }
                _ => {}
            }
        }
        if have_edge_schema {
            reversed_direction_warnings(pattern, &var_label, graph, &mut seen, &mut reversed);
        }
    });

    // The absent-property findings (A1b + eval §3a) travel alongside the
    // unknown-label/rel ones even when the labels/rels are all valid — and
    // separately from them, because they are the only family a locked schema
    // rejects rather than reports.
    let absent_property = absent_property_findings(query, graph, &var_label);
    // Family 5 (declared-type mismatches) shares `var_label` and its
    // conservatisms, but never its dedup key — see [`super::type_mismatch`].
    let mut type_mismatch = super::type_mismatch::type_mismatch_findings(query, graph, &var_label);
    let mut out: Vec<String> = Vec::new();
    if unknown_labels.is_empty() && unknown_rels.is_empty() {
        out.append(&mut reversed);
        out.append(&mut type_mismatch);
        return QueryWarnings {
            absent_property,
            other: out,
        };
    }

    out.reserve(unknown_labels.len() + unknown_rels.len());
    if !unknown_labels.is_empty() {
        let candidates: Vec<&str> = graph
            .node_type_metadata
            .keys()
            .map(|s| s.as_str())
            .chain(graph.type_indices.keys())
            .collect();
        for label in &unknown_labels {
            out.push(format!(
                "MATCH references unknown node label '{label}' — the graph has no such type, \
                 so this pattern returns no rows.{}",
                did_you_mean(label, &candidates)
            ));
        }
    }
    if !unknown_rels.is_empty() {
        let candidates: Vec<&str> = graph
            .connection_type_metadata
            .keys()
            .map(|s| s.as_str())
            .collect();
        for (rel, surviving) in &unknown_rels {
            let hint = did_you_mean(rel, &candidates);
            out.push(if surviving.is_empty() {
                format!(
                    "MATCH references unknown relationship type '{rel}' — the graph has no such \
                     edge type, so this pattern returns no rows.{hint}"
                )
            } else {
                // The pattern is an alternation with a live branch, so the
                // no-rows claim would be false about the query's result.
                let named = surviving
                    .iter()
                    .map(|s| format!("'{s}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "MATCH references unknown relationship type '{rel}' — the graph has no such \
                     edge type, so that branch matches no edges; the pattern can still return \
                     rows via {named}.{hint}"
                )
            });
        }
    }
    out.append(&mut reversed);
    out.append(&mut type_mismatch);
    QueryWarnings {
        absent_property,
        other: out,
    }
}

/// Reject a locked schema's absent-property reads.
///
/// **Caller gates on `graph.schema_locked`.** `lock_schema()` is the opt-in
/// "catch my typos" mechanism, and before this it covered a typo asymmetrically:
/// `MATCH (p:Person {agee: 1})` and `MATCH (p:Persn)` both failed, while
/// `MATCH (p:Person) WHERE p.agee = 1` returned an empty result and
/// `RETURN p.agee` returned a column of nulls — the two shapes an LLM or a
/// hurried human writes most. Same finding, same conservatism rules
/// ([`absent_property_findings`]); only the disposition differs.
///
/// Returns the **first** violation, like [`validate_schema`], and reuses the
/// unknown-property wording from [`validate_property`] so a locked schema
/// reports the same mistake identically whether it is written in a pattern
/// literal or in an expression.
///
/// One finding is deliberately *not* promoted: a property the graph's
/// [`schema_definition`](crate::graph::schema::DirGraph::schema_definition)
/// declares but no node has written yet. The all-null column is real, so the
/// warning stands, but the name is not a typo — it is in the user's own
/// declared model, and the pattern-literal check accepts it for exactly that
/// reason ([`property_is_declared`](super::property_is_declared)). Rejecting it
/// would make a lock disagree with the schema it locks.
pub(crate) fn strict_read_error(
    findings: &[AbsentProperty],
    graph: &DirGraph,
) -> Option<SchemaError> {
    let found = findings
        .iter()
        .find(|f| !super::property_is_declared(&f.label, &f.property, graph))?;
    let mut valid: Vec<&str> = graph
        .node_type_metadata
        .get(&found.label)
        .map(|m| m.keys().map(|s| s.as_str()).collect())
        .unwrap_or_default();
    valid.sort_unstable();
    Some(SchemaError {
        kind: SchemaErrorKind::UnknownProperty,
        message: format!(
            "Unknown property '{}' on {}, referenced in {}.{}\n  Valid properties: {}\n  \
             (the schema is locked — call unlock_schema() to make this a warning instead)",
            found.property,
            found.label,
            found.site.clause(),
            found.hint,
            valid.join(", ")
        ),
    })
}

/// Where the *echo* of a query warning goes.
///
/// Only the echo. The structured channel — [`QueryDiagnostics::warnings`], and
/// everything built on it (`ResultView.diagnostics`, the MCP `warnings:`
/// block) — is unconditional and no variant here can switch it off: the
/// computation happens once either way, and a caller that reads the field must
/// never depend on a process-global setting for it.
///
/// A host binding that wants its own presentation (the Python wheel's
/// `pywarn` policy re-emits through `warnings.warn`) selects [`Self::Silent`]
/// here and does its own emission from the diagnostics it already holds — the
/// engine never calls back into a host language.
///
/// [`QueryDiagnostics::warnings`]: crate::graph::languages::cypher::result::QueryDiagnostics::warnings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryWarningSink {
    /// Print `warning: <msg>` to stderr. The default, and what every release
    /// before the sink existed did unconditionally.
    #[default]
    Stderr,
    /// Print nothing. The structured channel is unaffected.
    Silent,
}

const SINK_STDERR: u8 = 0;
const SINK_SILENT: u8 = 1;

/// Process-global, because the emit sites are deep inside the executor and
/// the plan-cache path, and threading a presentation choice through
/// `ExecuteOptions` would put it on every binding's hot argument struct for a
/// setting no caller varies per query.
static SINK: AtomicU8 = AtomicU8::new(SINK_STDERR);

/// Select where query-warning echoes go, process-wide. Returns nothing; read
/// the current value back with [`query_warning_sink`].
pub fn set_query_warning_sink(sink: QueryWarningSink) {
    SINK.store(
        match sink {
            QueryWarningSink::Stderr => SINK_STDERR,
            QueryWarningSink::Silent => SINK_SILENT,
        },
        Ordering::Relaxed,
    );
}

/// The sink [`emit_query_warnings`] is currently writing to.
pub fn query_warning_sink() -> QueryWarningSink {
    match SINK.load(Ordering::Relaxed) {
        SINK_SILENT => QueryWarningSink::Silent,
        _ => QueryWarningSink::Stderr,
    }
}

/// Emit already-collected query warnings to stderr, matching kglite's
/// `warning:`-prefixed convention for non-fatal query/load issues.
///
/// The one emitter, for the one computation: `session::execute` collects the
/// warnings once, hands them to [`QueryDiagnostics::warnings`] for every
/// programmatic surface, and passes the same slice through here for the
/// interactive one (CLI/REPL users read stderr). Anything that computes a
/// query warning of its own — `executor::call_clause`'s procedure-scope
/// checks — emits through here too, so the prefix never drifts.
///
/// Being the one emitter is also what makes [`set_query_warning_sink`] a
/// single-point switch: silencing the echo is a check here, not a flag
/// threaded through four call sites.
///
/// [`QueryDiagnostics::warnings`]: crate::graph::languages::cypher::result::QueryDiagnostics::warnings
pub(crate) fn emit_query_warnings(warnings: &[String]) {
    if query_warning_sink() == QueryWarningSink::Silent {
        return;
    }
    for msg in warnings {
        #[cfg(test)]
        echo_recorder::record(msg);
        eprintln!("warning: {msg}");
    }
}

/// In-crate observer for the stderr branch above. `eprintln!` is captured by
/// libtest and unreadable from a unit test, so without this the "silent really
/// suppresses" assertion would be vacuous — it could only check that the call
/// returned.
#[cfg(test)]
mod echo_recorder {
    use std::sync::Mutex;

    static RECORDED: Mutex<Vec<String>> = Mutex::new(Vec::new());

    pub(super) fn record(msg: &str) {
        RECORDED
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(msg.to_string());
    }

    /// Every echo recorded so far that contains `needle`. Filtered rather than
    /// drained: the crate's tests run in parallel and other queries echo into
    /// the same buffer.
    pub(super) fn matching(needle: &str) -> Vec<String> {
        RECORDED
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|m| m.contains(needle))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::graph_with_schema;
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;

    // ── A1b: WHERE-clause absent-property warnings (non-fatal) ──────────────

    #[test]
    fn warns_on_where_property_absent_from_label() {
        let g = graph_with_schema();
        // `is_external` is not a Person property → the comparison is always
        // null/false and filters everything; warn (non-fatal) + did-you-mean.
        let q = parse_cypher("MATCH (p:Person) WHERE p.is_external = false RETURN p").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].contains("is_external") && w[0].contains("Person"),
            "{}",
            w[0]
        );
        // A near-miss still gets a suggestion.
        let q2 = parse_cypher("MATCH (p:Person) WHERE p.agee = 1 RETURN p").unwrap();
        let w2 = collect_unknown_pattern_warnings(&q2, &g);
        assert!(
            w2.iter().any(|m| m.contains("Did you mean 'age'")),
            "{w2:?}"
        );
    }

    #[test]
    fn no_warning_on_present_or_builtin_property() {
        let g = graph_with_schema();
        // Declared property → no warning.
        let q = parse_cypher("MATCH (p:Person) WHERE p.age = 30 RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
        // Built-in field → no warning.
        let q2 = parse_cypher("MATCH (p:Person) WHERE p.id = 1 RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&q2, &g).is_empty());
    }

    #[test]
    fn no_warning_on_untyped_var() {
        let g = graph_with_schema();
        // No label on the var → can't reason about its properties → no warning
        // (avoids false positives on dynamically-typed graphs).
        let q = parse_cypher("MATCH (n) WHERE n.whatever = 1 RETURN n").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }

    // ── D2p §3a: projection absent-property warnings ───────────────────────
    //
    // The WHERE walk above answers "why did my filter drop every row?". The
    // projection walk answers the quieter twin: `RETURN v.imo` on a type that
    // has no `imo` yields a full column of nulls, and because the sibling
    // `v.name` title-aliases to a real value the rows read as half-correct.

    #[test]
    fn warns_on_return_projection_of_absent_property() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (p:Person) RETURN p.name, p.imo").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].starts_with("RETURN projects property 'imo'")
                && w[0].contains("no Person node has")
                && w[0].contains("null"),
            "{}",
            w[0]
        );
    }

    #[test]
    fn warns_on_with_projection_of_absent_property() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (p:Person) WITH p.imo AS x RETURN x").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].starts_with("WITH projects property 'imo'"), "{}", w[0]);
    }

    #[test]
    fn warns_on_order_by_absent_property() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (p:Person) RETURN p ORDER BY p.imo").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].starts_with("ORDER BY sorts on property 'imo'"),
            "{}",
            w[0]
        );
    }

    #[test]
    fn projection_warning_suggests_a_near_miss() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (p:Person) RETURN p.agee").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert!(w.iter().any(|m| m.contains("Did you mean 'age'")), "{w:?}");
    }

    #[test]
    fn no_projection_warning_for_present_or_builtin_property() {
        let g = graph_with_schema();
        // `email` is in the type's metadata — a *sparse* property (set on at
        // least one node) is exactly what metadata records, so it never warns
        // however many nodes leave it null.
        let q = parse_cypher("MATCH (p:Person) RETURN p.age, p.email, p.id, p.title").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }

    #[test]
    fn no_projection_warning_through_a_with_alias() {
        // Probe, pinned: `var_label` is built from MATCH node patterns only,
        // so a var re-bound by `WITH n AS m` is not label-typed and `m.bad`
        // draws no warning. This mirrors the WHERE walk exactly (same map);
        // extending aliasing tracking is deliberately out of scope here.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) WITH n AS m RETURN m.badprop").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
        // …and the same is true on the WHERE side, which is the behaviour
        // being mirrored.
        let q2 = parse_cypher("MATCH (n:Person) WITH n AS m WHERE m.badprop = 1 RETURN m").unwrap();
        assert!(collect_unknown_pattern_warnings(&q2, &g).is_empty());
    }

    #[test]
    fn no_warning_for_a_property_the_same_query_writes() {
        // Warnings are computed before execution, so a property this query is
        // about to create is *not* absent by the time it is projected.
        let g = graph_with_schema();
        for query in [
            "MATCH (p:Person) SET p.badprop = 1 RETURN p.badprop",
            "MATCH (p:Person) SET p += {badprop: 1} RETURN p.badprop",
            "MATCH (p:Person) SET p.badprop = 1 WITH p WHERE p.badprop = 1 RETURN p",
            "MATCH (p:Person) FOREACH (x IN [1] | SET p.badprop = x) RETURN p.badprop",
        ] {
            let q = parse_cypher(query).unwrap();
            let w = collect_unknown_pattern_warnings(&q, &g);
            assert!(w.is_empty(), "{query} -> {w:?}");
        }
    }

    #[test]
    fn projection_and_where_reference_warn_once() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (p:Person) WHERE p.imo = 1 RETURN p.imo").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        // The filter is the more consequential of the two, and it comes first
        // in the query, so it is the one reported.
        assert!(
            w[0].starts_with("WHERE references property 'imo'"),
            "{}",
            w[0]
        );
    }

    // ── Locked-schema promotion: which findings become errors ──────────────

    #[test]
    fn strict_read_error_reports_the_first_finding_in_error_voice() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (p:Person) WHERE p.agee = 1 RETURN p.imo").unwrap();
        let found = collect_query_warnings(&q, &g).absent_property;
        assert_eq!(found.len(), 2, "{found:?}");
        let err = strict_read_error(&found, &g).expect("both findings are typos");
        assert!(
            err.message
                .starts_with("Unknown property 'agee' on Person, referenced in WHERE."),
            "{}",
            err.message
        );
        assert!(
            err.message.contains("Did you mean 'age'?"),
            "{}",
            err.message
        );
        assert!(
            err.message.contains("Valid properties: age, email"),
            "{}",
            err.message
        );
        assert!(err.message.contains("unlock_schema()"), "{}", err.message);
        assert!(matches!(err.kind, SchemaErrorKind::UnknownProperty));
    }

    /// A property the graph's own `schema_definition` declares is not a typo,
    /// even though no node has written a value for it yet — the same carve-out
    /// `validate_property` makes for the pattern-literal form. The all-null
    /// column is real, so the *warning* stands; only the promotion is skipped,
    /// or a lock would reject the model it was asked to lock.
    #[test]
    fn a_declared_but_unwritten_property_warns_but_is_never_promoted() {
        use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

        let mut g = graph_with_schema();
        let mut declared = SchemaDefinition::default();
        declared.node_schemas.insert(
            "Person".to_string(),
            NodeSchemaDefinition {
                optional_fields: vec!["nickname".to_string()],
                ..Default::default()
            },
        );
        g.set_schema(declared, SchemaInstall::Replace)
            .expect("schema installs on an empty graph");

        let q = parse_cypher("MATCH (p:Person) RETURN p.nickname").unwrap();
        let collected = collect_query_warnings(&q, &g);
        assert_eq!(collected.absent_property.len(), 1);
        assert!(
            strict_read_error(&collected.absent_property, &g).is_none(),
            "a declared field must not be promoted to an error"
        );
        let messages = collected.into_messages();
        assert_eq!(messages.len(), 1, "{messages:?}");
        assert!(messages[0].contains("'nickname'"), "{}", messages[0]);

        // The undeclared sibling is still promoted, so the carve-out is a
        // carve-out and not a blanket exemption for the type.
        let q2 = parse_cypher("MATCH (p:Person) RETURN p.nicknmae").unwrap();
        let found2 = collect_query_warnings(&q2, &g).absent_property;
        assert!(strict_read_error(&found2, &g).is_some());
    }

    // ── D2p §3b: reversed relationship-direction warnings ──────────────────

    /// Directed schema: `AUTHORED` and `REVIEWED` run Person→Paper, `KNOWS`
    /// runs Person→Person, and `LINKS` has been observed in *both*
    /// orientations.
    fn graph_with_directed_schema() -> DirGraph {
        let mut g = graph_with_schema();
        g.upsert_connection_type_metadata("REVIEWED", "Person", "Paper", HashMap::new());
        g.upsert_connection_type_metadata("LINKS", "Person", "Paper", HashMap::new());
        g.upsert_connection_type_metadata("LINKS", "Paper", "Person", HashMap::new());
        g
    }

    #[test]
    fn warns_on_reversed_relationship_direction() {
        let g = graph_with_directed_schema();
        // AUTHORED runs Person→Paper; this pattern asks for Paper→Person.
        let q = parse_cypher("MATCH (a:Paper)-[:AUTHORED]->(p:Person) RETURN p").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        // Both orientations are named, so the reader can see which way to turn
        // the arrow without going back to describe().
        assert!(
            w[0].contains("'AUTHORED'")
                && w[0].contains("Paper → Person")
                && w[0].contains("Person → Paper")
                && w[0].contains("matches no edges"),
            "{}",
            w[0]
        );
    }

    #[test]
    fn warns_on_reversed_left_pointing_relationship() {
        let g = graph_with_directed_schema();
        // `<-` form: the traversal is Paper→Person again.
        let q = parse_cypher("MATCH (p:Person)<-[:AUTHORED]-(a:Paper) RETURN p").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("Paper → Person"), "{}", w[0]);
    }

    #[test]
    fn no_direction_warning_for_correct_orientation() {
        let g = graph_with_directed_schema();
        for query in [
            "MATCH (p:Person)-[:AUTHORED]->(a:Paper) RETURN p",
            "MATCH (a:Paper)<-[:AUTHORED]-(p:Person) RETURN p",
            "MATCH (p:Person)-[:KNOWS]->(q:Person) RETURN p",
        ] {
            let q = parse_cypher(query).unwrap();
            assert!(
                collect_unknown_pattern_warnings(&q, &g).is_empty(),
                "{query}"
            );
        }
    }

    #[test]
    fn no_direction_warning_for_undirected_pattern() {
        let g = graph_with_directed_schema();
        let q = parse_cypher("MATCH (a:Paper)-[:AUTHORED]-(p:Person) RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }

    #[test]
    fn no_direction_warning_when_an_endpoint_label_is_unknown() {
        let g = graph_with_directed_schema();
        // Nothing to test set membership against on the bare endpoint.
        let q = parse_cypher("MATCH (a:Paper)-[:AUTHORED]->(x) RETURN x").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }

    #[test]
    fn direction_warning_uses_a_match_bound_label_for_a_bare_endpoint() {
        let g = graph_with_directed_schema();
        // `x` carries no label *here*, but the same MATCH labelled it — the
        // var→label map the property walk builds answers for it.
        let q = parse_cypher("MATCH (x:Person) MATCH (a:Paper)-[:AUTHORED]->(x) RETURN x").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("'AUTHORED'"), "{}", w[0]);
    }

    #[test]
    fn no_direction_warning_when_both_orientations_are_recorded() {
        // `source_types`/`target_types` are unions over every observed pair, so
        // a type seen as both Person→Paper and Paper→Person reports both labels
        // in both sets and the forward test passes. That is the documented
        // imprecision: a multi-pair relationship type can hide a genuinely
        // reversed arrow (false negative), and never invents one (the check
        // only fires when the forward union has no support at all).
        let g = graph_with_directed_schema();
        for query in [
            "MATCH (p:Person)-[:LINKS]->(a:Paper) RETURN p",
            "MATCH (a:Paper)-[:LINKS]->(p:Person) RETURN p",
        ] {
            let q = parse_cypher(query).unwrap();
            assert!(
                collect_unknown_pattern_warnings(&q, &g).is_empty(),
                "{query}"
            );
        }
    }

    #[test]
    fn alternation_warns_only_when_every_branch_is_reversed() {
        let g = graph_with_directed_schema();
        // KNOWS is Person→Person, so a Paper→Person traversal is not "reversed"
        // for it — the pattern could still match through that branch. Silence.
        let mixed = parse_cypher("MATCH (a:Paper)-[:AUTHORED|KNOWS]->(p:Person) RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&mixed, &g).is_empty());
        // Every branch reversed → the pattern really does match nothing.
        let all =
            parse_cypher("MATCH (a:Paper)-[:AUTHORED|REVIEWED]->(p:Person) RETURN p").unwrap();
        let w = collect_unknown_pattern_warnings(&all, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].contains("'AUTHORED'") && w[0].contains("'REVIEWED'"),
            "{}",
            w[0]
        );
    }

    #[test]
    fn no_direction_warning_for_a_variable_length_edge() {
        let g = graph_with_directed_schema();
        // The endpoint labels of a multi-hop expansion are not the endpoint
        // types of a single stored edge, so the membership test does not apply.
        let q = parse_cypher("MATCH (a:Paper)-[:AUTHORED*1..3]->(p:Person) RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }

    #[test]
    fn direction_warning_reaches_nested_patterns() {
        let g = graph_with_directed_schema();
        let q = parse_cypher(
            "MATCH (p:Person) WHERE EXISTS { MATCH (a:Paper)-[:AUTHORED]->(p) } RETURN p",
        )
        .unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("'AUTHORED'"), "{}", w[0]);
    }

    #[test]
    fn no_direction_warning_for_an_unknown_relationship_type() {
        // The unknown-rel-type warning already covers it; a second message
        // about orientation would be noise about an edge type that has none.
        let g = graph_with_directed_schema();
        let q = parse_cypher("MATCH (a:Paper)-[:AUTHORD]->(p:Person) RETURN p").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("unknown relationship type"), "{}", w[0]);
    }

    // ── the warning-echo sink ───────────────────────────────────────────────

    /// The sink is process-global; two tests flipping it at once would race.
    static SINK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn stderr_sink_echoes_and_silent_sink_does_not() {
        let _guard = SINK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Unique per assertion, so a parallel test echoing into the shared
        // recorder cannot make either half pass or fail by accident.
        let echoed = "sink-probe-echoed-a7f3".to_string();
        let muted = "sink-probe-muted-a7f3".to_string();

        assert_eq!(query_warning_sink(), QueryWarningSink::Stderr, "default");
        emit_query_warnings(std::slice::from_ref(&echoed));
        assert_eq!(echo_recorder::matching(&echoed).len(), 1);

        set_query_warning_sink(QueryWarningSink::Silent);
        assert_eq!(query_warning_sink(), QueryWarningSink::Silent);
        emit_query_warnings(std::slice::from_ref(&muted));
        assert!(
            echo_recorder::matching(&muted).is_empty(),
            "silent sink still echoed"
        );

        // Restore: every other test in the crate assumes the default.
        set_query_warning_sink(QueryWarningSink::Stderr);
        emit_query_warnings(std::slice::from_ref(&muted));
        assert_eq!(
            echo_recorder::matching(&muted).len(),
            1,
            "sink did not restore"
        );
    }
}
