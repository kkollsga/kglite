//! A Cypher write of ISO text onto a declared validity bound stores a date or
//! datetime, as the loaders do (`coerce_bounds.rs`): `CREATE`, `SET`, map
//! `SET`, `MERGE` with `ON CREATE` / `ON MATCH`, on nodes and relationships,
//! in every storage mode. Text that does not parse is refused.

use std::collections::HashMap;

use chrono::NaiveDate;

use super::declarations::{declare, TemporalTarget};
use super::eval::IntervalConvention::Closed;
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

fn day(text: &str) -> Value {
    Value::DateTime(NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap())
}

fn run(graph: &mut DirGraph, query: &str) -> Result<(), String> {
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(graph, query, &ExecuteOptions::eager(&params))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Property `property` of the `Status` node `id`, read from storage (a
/// Cypher read would be filtered to rows valid today).
fn status(graph: &DirGraph, id: i64, property: &str) -> Value {
    use crate::graph::storage::GraphRead;
    let idx = graph
        .type_indices
        .get("Status")
        .expect("Status nodes")
        .iter()
        .find(|&idx| {
            graph.graph.get_node_id(idx).is_some_and(|found| {
                found == Value::Int64(id) || found == Value::UniqueId(id as u32)
            })
        })
        .unwrap_or_else(|| panic!("no Status {id}"));
    super::node_bound(graph, idx, property)
}

/// Property `property` of the one `OP` relationship whose `vf` is `from`.
fn operation(graph: &DirGraph, from: &str, property: &str) -> Value {
    use crate::graph::schema::InternedKey;
    use crate::graph::storage::GraphRead;
    let hits: Vec<Value> = graph
        .graph
        .edge_indices()
        .filter(|&e| super::edge_bound(graph, e, InternedKey::from_str("vf")) == day(from))
        .map(|e| super::edge_bound(graph, e, InternedKey::from_str(property)))
        .collect();
    assert_eq!(hits.len(), 1, "{hits:?}");
    hits[0].clone()
}

/// A graph in `mode` with one declared `Status` node and one declared `OP`
/// relationship, both carrying real dates.
fn declared(mode: StorageMode, dir: &tempfile::TempDir) -> DirGraph {
    let path = matches!(mode, StorageMode::Disk).then(|| dir.path());
    let mut graph = new_dir_graph_in_mode(mode, path).expect("graph");
    run(
        &mut graph,
        "CREATE (:Status {id: 1, vf: date('2000-01-01'), vt: date('2010-01-01')}), \
         (:Project {id: 1})-[:OP {vf: date('2000-01-01'), vt: date('2001-01-01')}]->(:Co {id: 2})",
    )
    .unwrap();
    declare(
        &mut graph,
        &TemporalTarget::Node("Status".into()),
        "vf",
        "vt",
        Closed,
    )
    .unwrap();
    declare(
        &mut graph,
        &TemporalTarget::Relationship {
            rel_type: "OP".into(),
            source_type: None,
        },
        "vf",
        "vt",
        Closed,
    )
    .unwrap();
    graph
}

fn each_mode(check: impl Fn(StorageMode, DirGraph)) {
    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let dir = tempfile::tempdir().unwrap();
        check(mode, declared(mode, &dir));
    }
}

#[test]
fn a_create_of_iso_text_onto_a_bound_stores_dates() {
    each_mode(|mode, mut graph| {
        run(
            &mut graph,
            "CREATE (:Status {id: 2, vf: '2011-01-01', vt: '2012-01-01'})",
        )
        .unwrap();
        assert_eq!(status(&graph, 2, "vf"), day("2011-01-01"), "{mode:?}");
        assert_eq!(status(&graph, 2, "vt"), day("2012-01-01"), "{mode:?}");
        run(
            &mut graph,
            "MATCH (s:Status {id: 2}) SET s.note = '2020-01-01'",
        )
        .unwrap();
        assert_eq!(
            status(&graph, 2, "note"),
            Value::String("2020-01-01".into()),
            "{mode:?}: an undeclared property stays text"
        );
    });
}

#[test]
fn a_datetime_text_becomes_a_timestamp() {
    each_mode(|mode, mut graph| {
        run(
            &mut graph,
            "CREATE (:Status {id: 2, vf: '2011-01-01T10:30:00'})",
        )
        .unwrap();
        assert!(
            matches!(status(&graph, 2, "vf"), Value::Timestamp(_)),
            "{mode:?}"
        );
    });
}

#[test]
fn set_and_map_set_of_iso_text_store_dates() {
    each_mode(|mode, mut graph| {
        run(
            &mut graph,
            "MATCH (s:Status {id: 1}) SET s.vt = '2020-01-01'",
        )
        .unwrap();
        assert_eq!(status(&graph, 1, "vt"), day("2020-01-01"), "{mode:?}");
        run(
            &mut graph,
            "MATCH (s:Status {id: 1}) SET s += {vf: '1999-01-01'}",
        )
        .unwrap();
        assert_eq!(status(&graph, 1, "vf"), day("1999-01-01"), "{mode:?}");
        run(
            &mut graph,
            "MATCH (s:Status {id: 1}) SET s += {vf: '1998-01-01', vt: '2021-01-01'}",
        )
        .unwrap();
        assert_eq!(status(&graph, 1, "vf"), day("1998-01-01"), "{mode:?}");
    });
}

#[test]
fn merge_create_and_match_branches_store_dates() {
    each_mode(|mode, mut graph| {
        let merge = "MERGE (s:Status {id: 3}) ON CREATE SET s.vf = '2013-01-01' \
                     ON MATCH SET s.vt = '2014-01-01'";
        run(&mut graph, merge).unwrap();
        assert_eq!(status(&graph, 3, "vf"), day("2013-01-01"), "{mode:?}");
        run(&mut graph, merge).unwrap();
        assert_eq!(status(&graph, 3, "vt"), day("2014-01-01"), "{mode:?}");
        run(&mut graph, "MERGE (:Status {id: 4, vf: '2015-01-01'})").unwrap();
        assert_eq!(status(&graph, 4, "vf"), day("2015-01-01"), "{mode:?}");
    });
}

#[test]
fn a_relationship_bound_written_as_text_is_stored_as_a_date() {
    each_mode(|mode, mut graph| {
        run(
            &mut graph,
            "MATCH (p:Project {id: 1}), (c:Co {id: 2}) \
             CREATE (p)-[:OP {vf: '2003-01-01', vt: '2004-01-01'}]->(c)",
        )
        .unwrap();
        assert_eq!(
            operation(&graph, "2003-01-01", "vt"),
            day("2004-01-01"),
            "{mode:?}"
        );
        run(
            &mut graph,
            "MATCH (:Project)-[r:OP]->(:Co) WHERE r.vf = date('2003-01-01') SET r.vt = '2005-01-01'",
        )
        .unwrap();
        assert_eq!(
            operation(&graph, "2003-01-01", "vt"),
            day("2005-01-01"),
            "{mode:?}"
        );
    });
}

#[test]
fn unparsable_text_on_a_bound_is_refused_and_leaves_the_graph_alone() {
    each_mode(|mode, mut graph| {
        let err = run(
            &mut graph,
            "MATCH (s:Status {id: 1}) SET s.vt = 'not a date'",
        )
        .unwrap_err();
        assert!(err.contains("vt"), "{mode:?}: {err}");
        assert_eq!(status(&graph, 1, "vt"), day("2010-01-01"), "{mode:?}");
        assert!(
            run(&mut graph, "CREATE (:Status {id: 9, vf: 'nope'})").is_err(),
            "{mode:?}"
        );
    });
}
