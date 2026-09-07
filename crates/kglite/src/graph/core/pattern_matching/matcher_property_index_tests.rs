//! The **property**-index miss contract and the ontology closure probe.
//!
//! Sibling of `matcher_id_lookup_tests.rs`, which pins the same distinction
//! for the id anchor. The shape under test is again the *return shape* of
//! [`PatternExecutor::try_index_lookup`], because that is what
//! `find_matching_nodes` reads as "answered" vs "scan the whole type":
//!
//! * `Some(v)` — the index answered; `v` is used verbatim.
//! * `None`    — nothing covers the pattern; the caller scans the type.
//!
//! So `Some(vec![])` for a value with no bucket *is* the assertion that no
//! scan happens, and it is only sound where the index covers the same
//! value-space a scan would read — see `index_answers_point_lookup`.

use super::*;
use crate::graph::ontology::ontology_from_json;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

fn rows(graph: &DirGraph, query: &str) -> usize {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("read query failed: {query}: {e}"))
        .result
        .rows
        .len()
}

fn eq_props(name: &str, value: Value) -> HashMap<String, PropertyMatcher> {
    HashMap::from([(name.to_string(), PropertyMatcher::Equals(value))])
}

fn s(value: &str) -> Value {
    Value::String(value.to_string())
}

fn lookup(
    graph: &DirGraph,
    node_type: &str,
    props: &HashMap<String, PropertyMatcher>,
) -> Option<Vec<NodeIndex>> {
    PatternExecutor::new(graph, None).try_index_lookup(node_type, props)
}

fn closure_probe(
    graph: &DirGraph,
    node_type: &str,
    props: &HashMap<String, PropertyMatcher>,
) -> Option<Vec<NodeIndex>> {
    PatternExecutor::new(graph, None).try_closure_index_lookup(node_type, props)
}

/// Two Students and one Teacher under an abstract `Person`, materialized so
/// `:Person` is a **Closed** managed label. `email` is a plain stored
/// property — deliberately not `name`/`title`, whose index contents are not
/// the value-space a scan reads (see the soft-alias tests below).
fn school(index_email_on: &[&str]) -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Student {id: 1, name: 'Ann', email: 'ann@x'}), \
         (:Student {id: 2, name: 'Bo', email: 'bo@x'})",
    );
    run(
        &mut graph,
        "CREATE (:Teacher {id: 10, name: 'Tea', email: 'tea@x'})",
    );
    let store = ontology_from_json(
        r#"{"classes": {"Person": {"abstract": true},
                        "Student": {"is_a": "Person"},
                        "Teacher": {"is_a": "Person"}}}"#,
    )
    .expect("ontology parses");
    graph.define_ontology(store).expect("ontology installs");
    for node_type in index_email_on {
        graph.create_index(node_type, "email");
    }
    graph
        .materialize_ontology(false)
        .expect("materialization succeeds");
    assert!(
        graph.managed_label_closed("Person"),
        "precondition: :Person must be a Closed managed label"
    );
    graph
}

/// **The single-type fix.** A value with no bucket in a built index is
/// proven empty, not an unbuilt index: falling through re-derives the same
/// empty answer at the cost of a full type scan.
#[test]
fn absent_value_with_an_index_answers_empty_without_scanning() {
    let graph = school(&["Student"]);
    assert_eq!(
        lookup(&graph, "Student", &eq_props("email", s("nobody@x"))),
        Some(Vec::new()),
        "a value with no bucket in a built index must answer empty, not fall through to a scan"
    );
    // The hit path is untouched.
    assert_eq!(
        lookup(&graph, "Student", &eq_props("email", s("bo@x")))
            .expect("present value answers")
            .len(),
        1
    );
    // …and no index at all is still a scan.
    assert_eq!(
        lookup(&graph, "Teacher", &eq_props("email", s("tea@x"))),
        None,
        "an unindexed (type, property) must still fall through to the scan"
    );
}

/// **The closure fix.** A unique value lives in at most *one* member's
/// index, so a union that declines on any member's value-miss can never
/// fire with two live members. The probe must treat a covered miss as an
/// empty contribution.
#[test]
fn closure_probe_unions_members_when_the_value_lives_in_one() {
    let graph = school(&["Student", "Teacher"]);
    let hit = closure_probe(&graph, "Person", &eq_props("email", s("tea@x")))
        .expect("every member is indexed: the probe must answer");
    assert_eq!(hit.len(), 1, "exactly the Teacher row");
    assert_eq!(
        graph.graph.get_node_id(hit[0]),
        Some(Value::Int64(10)),
        "the probe must return the Teacher that holds the value"
    );

    // The other member's value, and a value in neither.
    assert_eq!(
        closure_probe(&graph, "Person", &eq_props("email", s("ann@x")))
            .expect("the probe answers")
            .len(),
        1
    );
    assert_eq!(
        closure_probe(&graph, "Person", &eq_props("email", s("nobody@x"))),
        Some(Vec::new())
    );

    // End-to-end, the answers a scan would have given.
    assert_eq!(
        rows(&graph, "MATCH (p:Person {email: 'tea@x'}) RETURN p.id"),
        1
    );
    assert_eq!(
        rows(&graph, "MATCH (p:Person {email: 'ann@x'}) RETURN p.id"),
        1
    );
    assert_eq!(
        rows(&graph, "MATCH (p:Person {email: 'nobody@x'}) RETURN p.id"),
        0
    );
}

/// Partial coverage stays a wholesale decline: a union missing a member's
/// rows would silently drop them.
#[test]
fn closure_probe_declines_when_a_member_is_unindexed() {
    let graph = school(&["Student"]);
    assert_eq!(
        closure_probe(&graph, "Person", &eq_props("email", s("tea@x"))),
        None,
        "Teacher carries no index: the probe must decline rather than drop its rows"
    );
    assert_eq!(
        rows(&graph, "MATCH (p:Person {email: 'tea@x'}) RETURN p.id"),
        1,
        "and the scan must still find the Teacher"
    );
}

/// `name` resolves through the **structural soft-alias fallback** (a node
/// with no stored `name` answers with its title), but `create_index` builds
/// only from the stored property. The index is therefore a *subset* of what a
/// scan matches, and must never be read as authoritative — neither for a
/// proven-empty nor for a closure union.
#[test]
fn a_soft_alias_index_never_proves_a_miss_or_covers_a_closure() {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:T {id: 1, name: 'Ann'})");
    // No stored `name`; `n.name` still resolves to 'Ann' through the title.
    run(&mut graph, "CREATE (:T {id: 2, title: 'Ann'})");
    assert_eq!(
        rows(&graph, "MATCH (n:T {name: 'Ann'}) RETURN n.id"),
        2,
        "precondition: both nodes match before any index exists"
    );

    graph.create_index("T", "name");
    assert_eq!(
        rows(&graph, "MATCH (n:T {name: 'Ann'}) RETURN n.id"),
        2,
        "creating an index must not change the answer"
    );
    assert_eq!(
        lookup(&graph, "T", &eq_props("name", s("absent"))),
        None,
        "a soft-alias index cannot prove a miss: the scan sees values it never indexed"
    );
}

/// The sound per-type alias family is the **registered** id/title alias, not
/// the `title`/`name`/`label` spelling family: an index registered under a
/// type's title-alias spelling holds titles (`create_index`'s own contract),
/// so a query written as `{title: …}` is served by it and vice versa.
#[test]
fn a_registered_title_alias_index_serves_the_canonical_spelling() {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:Term {id: 1, term_name: 'alpha'})");
    graph
        .title_field_aliases_mut()
        .insert("Term".to_string(), "term_name".to_string());
    // Rebuild under the alias so the node's title is the aliased column.
    run(&mut graph, "MATCH (n:Term {id: 1}) SET n.title = 'alpha'");
    graph.create_index("Term", "term_name");

    assert_eq!(
        lookup(&graph, "Term", &eq_props("title", s("alpha")))
            .expect("the alias-registered index answers the canonical spelling")
            .len(),
        1
    );
    assert_eq!(
        lookup(&graph, "Term", &eq_props("title", s("absent"))),
        Some(Vec::new()),
        "and it proves a miss, being over the same field"
    );
}

/// The **disk** mirror of `a_soft_alias_index_never_proves_a_miss_or_covers_a
/// _closure`. A persistent bundle is built from stored values exactly like the
/// in-memory map, so reading one on a structurally-resolved name drops the
/// rows that resolve through the title or the type string.
///
/// Deep-scan T2-6b: before this, `create_index('C', 'label')` turned
/// `WHERE n.label = 'C'` from one row into zero on a disk graph — a wrong
/// answer no reporting surface could have warned about, because the query
/// never consulted the reporting.
#[test]
fn a_soft_alias_disk_bundle_never_answers_a_lookup() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:C {id: 1, label: 'Norway'})");
    run(&mut graph, "CREATE (:C {id: 2, label: 'Sweden'})");
    // No stored `label`; `n.label` still resolves — to the type string 'C'.
    run(&mut graph, "CREATE (:C {id: 3})");
    graph.enable_disk_mode().unwrap();
    graph.save_disk(dir.path().to_str().unwrap()).unwrap();

    let probes = [
        "MATCH (n:C) WHERE n.label = 'C' RETURN n.id",
        "MATCH (n:C {label: 'C'}) RETURN n.id",
        "MATCH (n:C) WHERE n.label STARTS WITH 'C' RETURN n.id",
    ];
    for query in probes {
        assert_eq!(rows(&graph, query), 1, "precondition, no index: {query}");
    }

    let (_unique, persistent) = graph.create_property_index_routed("C", "label").unwrap();
    assert!(persistent, "a disk create_index builds the mmap bundle");

    for query in probes {
        assert_eq!(
            rows(&graph, query),
            1,
            "creating a disk index must not change the answer: {query}"
        );
    }
    assert_eq!(
        rows(&graph, "MATCH (n:C {label: 'Norway'}) RETURN n.id"),
        1,
        "and the stored-value rows are still found by the scan"
    );
    assert!(
        !graph.index_serves_lookups("C", "label"),
        "a bundle no lookup reads must not be reported as serving"
    );
    assert!(
        graph.index_not_serving_reason("C", "label").is_some(),
        "and the caller is owed the reason"
    );
}

/// The cross-type arms consult a bundle named by an *alias family member*, so
/// the exclusion has to test the bundle's name as well as the queried
/// property: a global `name` bundle holds stored `name` values, and answering
/// `{title: 'X'}` from it drops every node whose title is not also a stored
/// `name`.
///
/// Reachable whenever the `name` bundle is the fresher one — here the `title`
/// global that every disk `save()` builds has declined under a later write
/// (see `index_freshness_tests`), so the alias loop falls through to `name`.
#[test]
fn a_stored_name_global_bundle_never_answers_a_title_lookup() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:T {id: 1, title: 'Alpha', name: 'Alpha'})",
    );
    graph.enable_disk_mode().unwrap();
    graph.save_disk(dir.path().to_str().unwrap()).unwrap();

    // Written after the save, so the auto-built `title` global declines and
    // the loop reaches the `name` bundle built next.
    run(&mut graph, "CREATE (:T {id: 2, title: 'Beta'})");
    match &mut graph.graph {
        crate::api::storage::GraphBackend::Disk(disk) => {
            disk.build_global_property_index("name").unwrap();
        }
        _ => panic!("disk backend"),
    }

    assert_eq!(
        rows(&graph, "MATCH (n {title: 'Beta'}) RETURN n.id"),
        1,
        "the `name` bundle does not hold node 2's title and must not answer"
    );
    assert_eq!(
        rows(&graph, "MATCH (n:T) WHERE n.title = 'Beta' RETURN n.id"),
        1,
        "same through the typed cross-type arm"
    );
}
