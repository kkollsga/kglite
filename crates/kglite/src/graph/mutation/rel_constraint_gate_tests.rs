//! The per-conflict-mode matrix for the bulk relationship-constraint gate.
//!
//! Every mode merges a row into an existing edge differently, so "does this
//! row violate the constraint?" has a different answer in each — and two of
//! them are counter-intuitive: `Preserve` must **accept** a bad row value
//! (it is discarded), and `Sum` must **refuse** a pair of values that are each
//! individually fine (their sum is not).

use crate::datatypes::{DataFrame, Value};
use crate::graph::algorithms::Interrupt;
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::maintain::{add_connections, add_nodes};
use crate::graph::property_types::DeclaredType;
use crate::graph::storage::interner::InternedKey;
use crate::graph::storage::GraphRead;

fn nodes(graph: &mut DirGraph, node_type: &str, ids: Vec<Value>) {
    let frame = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        ids.into_iter().map(|id| vec![id]).collect(),
    )
    .unwrap();
    add_nodes(
        graph,
        frame,
        node_type.to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
}

/// A frame of `(source, target, property)` rows. `None` writes a null cell,
/// which the loader treats as an absent property — the same collapse the node
/// gate makes.
fn edges_df(column: &str, rows: &[(i64, &str, Option<Value>)]) -> DataFrame {
    DataFrame::from_cypher_rows(
        vec!["s".to_string(), "t".to_string(), column.to_string()],
        rows.iter()
            .map(|(source, target, value)| {
                vec![
                    Value::Int64(*source),
                    Value::String((*target).into()),
                    value.clone().unwrap_or(Value::Null),
                ]
            })
            .collect(),
    )
    .unwrap()
}

/// A frame carrying no property column at all — the partial-update shape.
fn bare_df(rows: &[(i64, &str)]) -> DataFrame {
    DataFrame::from_cypher_rows(
        vec!["s".to_string(), "t".to_string()],
        rows.iter()
            .map(|(source, target)| vec![Value::Int64(*source), Value::String((*target).into())])
            .collect(),
    )
    .unwrap()
}

fn mentions(graph: &mut DirGraph, frame: DataFrame, mode: Option<&str>) -> Result<(), String> {
    add_connections(
        graph,
        frame,
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Entity".to_string(),
        "t".to_string(),
        None,
        None,
        mode.map(str::to_string),
    )
    .map(|_| ())
}

/// Two `Doc`s, three `Entity`s, and one stored `MENTIONS` edge `(1, A)`
/// carrying `weight = 10` and `since = 2020` — installed *before* the
/// constraint, so the declaration scan vouches for it.
fn seeded_graph(constrain: impl Fn(&mut DirGraph)) -> DirGraph {
    let mut graph = DirGraph::new();
    nodes(&mut graph, "Doc", vec![Value::Int64(1), Value::Int64(2)]);
    nodes(
        &mut graph,
        "Entity",
        vec![
            Value::String("A".into()),
            Value::String("B".into()),
            Value::String("C".into()),
        ],
    );
    let seed = DataFrame::from_cypher_rows(
        vec![
            "s".to_string(),
            "t".to_string(),
            "weight".to_string(),
            "since".to_string(),
        ],
        vec![vec![
            Value::Int64(1),
            Value::String("A".into()),
            Value::Int64(10),
            Value::Int64(2020),
        ]],
    )
    .unwrap();
    mentions(&mut graph, seed, None).expect("seed edge");
    constrain(&mut graph);
    graph
}

fn typed_graph() -> DirGraph {
    seeded_graph(|graph| {
        graph
            .create_rel_property_type_constraint(
                "MENTIONS",
                "weight",
                DeclaredType::Integer,
                &Interrupt::default(),
            )
            .expect("declaration must install over clean data");
    })
}

fn required_graph() -> DirGraph {
    seeded_graph(|graph| {
        graph
            .create_rel_not_null_constraint("MENTIONS", "since", &Interrupt::default())
            .expect("declaration must install over clean data");
    })
}

fn edge_count(graph: &DirGraph, source: i64) -> usize {
    let idx = graph
        .lookup_by_id_readonly("Doc", &Value::Int64(source))
        .unwrap();
    let key = InternedKey::from_str("MENTIONS");
    graph
        .graph
        .edges_directed(idx, petgraph::Direction::Outgoing)
        .filter(|edge| edge.connection_type() == key)
        .count()
}

fn stored_weight(graph: &DirGraph, source: i64) -> Option<Value> {
    let idx = graph
        .lookup_by_id_readonly("Doc", &Value::Int64(source))
        .unwrap();
    let key = InternedKey::from_str("MENTIONS");
    let weight = InternedKey::from_str("weight");
    graph
        .graph
        .edges_directed(idx, petgraph::Direction::Outgoing)
        .find(|edge| edge.connection_type() == key)
        .and_then(|edge| {
            edge.weight()
                .properties
                .iter()
                .find(|(k, _)| *k == weight)
                .map(|(_, v)| v.clone())
        })
}

// ── property type × conflict mode ────────────────────────────────────

/// Update: the row's value wins, so a bad one lands — and is refused.
#[test]
fn update_refuses_a_bad_row_value_over_a_good_stored_one() {
    let mut graph = typed_graph();
    let error = mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::String("heavy".into())))]),
        Some("update"),
    )
    .expect_err("the row's value wins under update, so it must be judged");
    assert!(error.contains("STRING"), "{error}");
    assert!(error.contains("MENTIONS.weight"), "{error}");
    assert_eq!(stored_weight(&graph, 1), Some(Value::Int64(10)));
}

/// Preserve: the stored value wins, so the row's bad value is *discarded*.
/// Refusing it would reject a write the engine never performs.
#[test]
fn preserve_accepts_a_bad_row_value_the_engine_will_discard() {
    let mut graph = typed_graph();
    mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::String("heavy".into())))]),
        Some("preserve"),
    )
    .expect("preserve keeps the stored value, so the row cannot violate anything");
    assert_eq!(stored_weight(&graph, 1), Some(Value::Int64(10)));
}

/// Preserve on a pair that does *not* exist yet is a create, and then the row
/// is the whole state — the same value must now be refused.
#[test]
fn preserve_refuses_the_same_bad_value_when_the_pair_is_new() {
    let mut graph = typed_graph();
    let error = mentions(
        &mut graph,
        edges_df("weight", &[(1, "B", Some(Value::String("heavy".into())))]),
        Some("preserve"),
    )
    .expect_err("nothing is stored for (1, B), so the row is the whole edge");
    assert!(error.contains("STRING"), "{error}");
    assert_eq!(edge_count(&graph, 1), 1, "the refused frame wrote nothing");
}

/// Replace drops the stored properties and rebuilds from the row.
#[test]
fn replace_refuses_a_bad_row_value() {
    let mut graph = typed_graph();
    let error = mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::String("heavy".into())))]),
        Some("replace"),
    )
    .expect_err("replace rebuilds the edge from the row alone");
    assert!(error.contains("STRING"), "{error}");
    assert_eq!(stored_weight(&graph, 1), Some(Value::Int64(10)));
}

/// Skip leaves an existing edge exactly as it was, so the row is judged
/// against nothing.
#[test]
fn skip_accepts_a_bad_row_value_for_an_existing_edge() {
    let mut graph = typed_graph();
    mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::String("heavy".into())))]),
        Some("skip"),
    )
    .expect("skip touches nothing, so nothing can violate a constraint");
    assert_eq!(stored_weight(&graph, 1), Some(Value::Int64(10)));
}

/// The mode that produces a value neither side wrote. `10` and `1.5` each
/// satisfy nothing and everything on their own; their sum is a FLOAT, and an
/// INTEGER declaration has to catch it. A gate that judged the row in
/// isolation would pass this.
#[test]
fn sum_refuses_a_row_whose_addition_changes_the_type() {
    let mut graph = typed_graph();
    let error = mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::Float64(1.5)))]),
        Some("sum"),
    )
    .expect_err("10 + 1.5 is a FLOAT, which the INTEGER declaration refuses");
    assert!(error.contains("FLOAT"), "{error}");
    assert_eq!(stored_weight(&graph, 1), Some(Value::Int64(10)));
}

/// The same mode must not refuse an addition that stays in type.
#[test]
fn sum_accepts_an_addition_that_stays_an_integer() {
    let mut graph = typed_graph();
    mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::Int64(5)))]),
        Some("sum"),
    )
    .expect("10 + 5 is an INTEGER");
    assert_eq!(stored_weight(&graph, 1), Some(Value::Int64(15)));
}

// ── presence × conflict mode ─────────────────────────────────────────

/// The partial-update contract: a frame that does not carry the required
/// column leaves the stored value alone, so it must not be refused. This is
/// the assertion that stops the gate from being written as "every row must
/// supply every required property".
#[test]
fn update_accepts_a_frame_that_does_not_carry_the_required_column() {
    let mut graph = required_graph();
    mentions(&mut graph, bare_df(&[(1, "A")]), Some("update"))
        .expect("the stored `since` survives an update that never mentions it");
}

/// The same frame against a pair with no stored edge creates one, and a
/// created edge has to satisfy the requirement itself.
#[test]
fn update_refuses_a_new_pair_with_no_value_for_the_required_property() {
    let mut graph = required_graph();
    let error = mentions(&mut graph, bare_df(&[(1, "B")]), Some("update"))
        .expect_err("a created edge must satisfy the requirement");
    assert!(error.contains("'since'"), "{error}");
    assert!(error.contains("relationship"), "{error}");
    assert_eq!(edge_count(&graph, 1), 1, "the refused frame wrote nothing");
}

/// Replace drops the stored value, so the row has to carry the required one.
#[test]
fn replace_refuses_a_row_without_the_required_property() {
    let mut graph = required_graph();
    let error = mentions(&mut graph, bare_df(&[(1, "A")]), Some("replace"))
        .expect_err("replace rebuilds from the row, which has no `since`");
    assert!(error.contains("'since'"), "{error}");
}

/// Preserve and Skip both keep the stored value, so neither can lose it.
#[test]
fn preserve_and_skip_keep_the_required_value() {
    for mode in ["preserve", "skip"] {
        let mut graph = required_graph();
        mentions(&mut graph, bare_df(&[(1, "A")]), Some(mode))
            .unwrap_or_else(|e| panic!("{mode}: {e}"));
    }
}

/// Within-frame consolidation: two rows for the same *new* pair land on one
/// edge, so the second row merges into the first's result rather than into
/// nothing. Judging each row alone would refuse the second one.
#[test]
fn a_second_row_for_the_same_new_pair_merges_into_the_first() {
    let mut graph = required_graph();
    mentions(
        &mut graph,
        edges_df(
            "since",
            &[(1, "B", Some(Value::Int64(2021))), (1, "B", None)],
        ),
        Some("update"),
    )
    .expect("the first row supplies `since`; the second merges onto that edge");
    assert_eq!(edge_count(&graph, 1), 2);
}

// ── the frame contract ───────────────────────────────────────────────

/// A refused frame is refused whole: no edge, no title write, and no
/// connection-type metadata for a type the frame failed to load. The loaders
/// validate-then-write, and a per-row skip here would fork that contract.
#[test]
fn a_refused_frame_writes_nothing_at_all() {
    let mut graph = required_graph();
    let before: Vec<String> = {
        let mut types: Vec<String> = graph.connection_type_metadata.keys().cloned().collect();
        types.sort();
        types
    };
    let error = mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::Int64(3))), (1, "C", None)]),
        Some("update"),
    )
    .expect_err("the second row creates an edge with no `since`");
    assert!(error.contains("'since'"), "{error}");
    assert_eq!(edge_count(&graph, 1), 1, "no edge from the good row either");
    let after: Vec<String> = {
        let mut types: Vec<String> = graph.connection_type_metadata.keys().cloned().collect();
        types.sort();
        types
    };
    assert_eq!(
        after, before,
        "a refused frame must not teach the schema anything"
    );
}

/// An unconstrained connection type pays the fast-out and nothing else: the
/// same frame that a constrained type refuses loads without complaint.
#[test]
fn an_unconstrained_connection_type_is_not_gated() {
    let mut graph = DirGraph::new();
    nodes(&mut graph, "Doc", vec![Value::Int64(1)]);
    nodes(&mut graph, "Entity", vec![Value::String("A".into())]);
    mentions(
        &mut graph,
        edges_df("weight", &[(1, "A", Some(Value::String("heavy".into())))]),
        Some("update"),
    )
    .expect("nothing is declared, so nothing is judged");
    assert_eq!(edge_count(&graph, 1), 1);
}

// ── the loader's two row-folding regimes ─────────────────────────────

/// **The regression this file exists to prevent.**
///
/// On an initial load of a new connection type the loader creates a
/// relationship *per row* — no lookup, no merge, no consolidation between two
/// rows naming the same pair. A gate that models rows as merging into one edge
/// per pair therefore judges a post-state the loader will never produce: it
/// sees `{since: 2020}` where the loader will store two relationships, the
/// second of them carrying no `since` at all.
///
/// That admits a stored relationship violating its own constraint, which
/// breaks the invariant `preserve` and `skip` lean on — that every stored edge
/// of a constrained type is already legal.
#[test]
fn an_initial_load_judges_each_row_as_its_own_relationship() {
    let mut graph = DirGraph::new();
    nodes(&mut graph, "Doc", vec![Value::Int64(1)]);
    nodes(&mut graph, "Entity", vec![Value::String("A".into())]);
    // Declared before any MENTIONS edge exists, so the type is absent from the
    // connection metadata and the load below takes the initial-load path.
    graph
        .create_rel_not_null_constraint("MENTIONS", "since", &Interrupt::default())
        .expect("an empty type is vacuously clean");

    let error = mentions(
        &mut graph,
        edges_df(
            "since",
            &[(1, "A", Some(Value::Int64(2020))), (1, "A", None)],
        ),
        Some("update"),
    )
    .expect_err("the second row becomes its own relationship, with no `since`");
    assert!(error.contains("'since'"), "{error}");
    assert_eq!(
        edge_count(&graph, 1),
        0,
        "the refused frame must store neither relationship"
    );
}

/// The same shape one load later. The type is now known, so the loader looks
/// edges up and the two rows consolidate onto one relationship — which the
/// first row made legal. Judging them independently here would refuse a write
/// the loader performs correctly.
#[test]
fn a_known_type_still_consolidates_two_rows_onto_one_relationship() {
    let mut graph = required_graph();
    mentions(
        &mut graph,
        edges_df(
            "since",
            &[(1, "B", Some(Value::Int64(2021))), (1, "B", None)],
        ),
        Some("update"),
    )
    .expect("the first row supplies `since`; the second merges onto that edge");
    assert_eq!(edge_count(&graph, 1), 2);
}
