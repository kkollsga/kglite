use super::*;
use crate::graph::dir_graph::index_layer::LayeredIndex;
use crate::graph::dir_graph::range_index_layer::LayeredRangeIndex;
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound::{Excluded, Included, Unbounded};

fn indexed(values: Vec<Value>) -> DirGraph {
    let mut graph = DirGraph::new();
    let mut hash = HashMap::new();
    let mut range = BTreeMap::new();
    for (slot, value) in values.into_iter().enumerate() {
        let nodes = vec![NodeIndex::new(slot)];
        hash.insert(value.clone(), nodes.clone());
        range.insert(value, nodes);
    }
    graph
        .property_indices
        .insert(("N".into(), "v".into()), LayeredIndex::from(hash));
    graph
        .range_indices
        .insert(("N".into(), "v".into()), LayeredRangeIndex::from(range));
    graph
}
fn slots(hits: Option<Vec<NodeIndex>>) -> Vec<usize> {
    let mut values: Vec<_> = hits
        .expect("the ordinary query must use the index")
        .into_iter()
        .map(NodeIndex::index)
        .collect();
    values.sort_unstable();
    values
}

#[test]
fn numeric_equality_index_admits_every_equivalent_variant() {
    let graph = indexed(vec![
        Value::UniqueId(1),
        Value::Int64(1),
        Value::Float64(1.0),
        Value::Int64(2),
    ]);
    for probe in [Value::UniqueId(1), Value::Int64(1), Value::Float64(1.0)] {
        assert_eq!(
            slots(graph.lookup_by_index("N", "v", &probe)),
            vec![0, 1, 2]
        );
    }
    assert_eq!(
        slots(graph.lookup_by_index("N", "v", &Value::Int64(9))),
        Vec::<usize>::new()
    );
    for probe in [
        Value::Int64(1 << 53),
        Value::Float64(f64::INFINITY),
        Value::List(vec![Value::Null]),
    ] {
        assert!(graph.lookup_by_index("N", "v", &probe).is_none());
    }
}

#[test]
fn numeric_range_translates_bounds_without_losing_variants() {
    let graph = indexed(vec![
        Value::UniqueId(1),
        Value::Int64(1),
        Value::Float64(1.0),
        Value::Float64(1.5),
        Value::Int64(2),
        Value::Null,
        Value::Float64(f64::NAN),
    ]);
    let one = Value::Float64(1.0);
    let two = Value::Int64(2);
    assert_eq!(
        slots(graph.lookup_range("N", "v", Included(&one), Included(&one))),
        vec![0, 1, 2]
    );
    assert_eq!(
        slots(graph.lookup_range("N", "v", Excluded(&one), Excluded(&two))),
        vec![3]
    );
    assert_eq!(
        slots(graph.lookup_range("N", "v", Unbounded, Included(&one))),
        vec![0, 1, 2, 5]
    );
    assert_eq!(
        slots(graph.lookup_range("N", "v", Excluded(&two), Unbounded)),
        Vec::<usize>::new()
    );
    assert_eq!(
        slots(graph.lookup_range("N", "v", Included(&two), Included(&one))),
        Vec::<usize>::new()
    );
    assert!(graph
        .lookup_range("N", "v", Included(&Value::Int64(1 << 53)), Unbounded)
        .is_none());
}

#[test]
fn string_index_expands_existing_wrapper_equivalence() {
    let graph = indexed(vec![
        Value::String("Oslo".into()),
        Value::String("[\"Oslo\"]".into()),
        Value::String("Bergen".into()),
    ]);
    for probe in ["Oslo", "[\"Oslo\"]"] {
        assert_eq!(
            slots(graph.lookup_by_index("N", "v", &Value::String(probe.into()))),
            vec![0, 1]
        );
    }
    let oslo = Value::String("Oslo".into());
    assert_eq!(
        slots(graph.lookup_range("N", "v", Included(&oslo), Included(&oslo))),
        vec![0]
    );
}

#[test]
fn soft_alias_ranges_decline_despite_stored_property_hits() {
    let mut graph = indexed(vec![Value::String("Ann".into())]);
    let index = graph
        .range_indices
        .remove(&("N".into(), "v".into()))
        .unwrap();
    graph
        .range_indices
        .insert(("N".into(), "name".into()), index);
    let lower = Value::String("A".into());
    assert!(graph
        .lookup_range("N", "name", Included(&lower), Unbounded)
        .is_none());
}

fn composite_indexed(properties: &[String], tuples: Vec<Vec<Value>>) -> DirGraph {
    let mut graph = DirGraph::new();
    let mut index = HashMap::new();
    for (slot, values) in tuples.into_iter().enumerate() {
        index
            .entry(CompositeValue(values))
            .or_insert_with(Vec::new)
            .push(NodeIndex::new(slot));
    }
    graph
        .composite_indices
        .insert(("N".into(), properties.to_vec()), LayeredIndex::from(index));
    graph
}

#[test]
fn composite_predicate_admits_all_numeric_tuples_and_keeps_raw_keys_exact() {
    let names = vec!["a".into(), "b".into()];
    let first = [Value::Int64(1), Value::Float64(1.0), Value::UniqueId(1)];
    let second = [Value::Int64(2), Value::Float64(2.0), Value::UniqueId(2)];
    let tuples = first
        .iter()
        .flat_map(|a| second.iter().map(move |b| vec![a.clone(), b.clone()]))
        .collect();
    let graph = composite_indexed(&names, tuples);
    for a in &first {
        for b in &second {
            assert_eq!(
                slots(graph.lookup_by_composite_predicate("N", &names, &[a.clone(), b.clone()])),
                (0..9).collect::<Vec<_>>()
            );
            assert_eq!(
                slots(graph.lookup_by_composite_predicate(
                    "N",
                    &["b".into(), "a".into()],
                    &[b.clone(), a.clone()]
                )),
                (0..9).collect::<Vec<_>>()
            );
        }
    }
    assert_eq!(
        slots(graph.lookup_by_composite_index("N", &names, &[first[0].clone(), second[0].clone()])),
        vec![0]
    );
    assert_eq!(
        slots(graph.lookup_by_composite_predicate(
            "N",
            &names,
            &[Value::Int64(9), Value::Int64(2)]
        )),
        Vec::<usize>::new()
    );
}

#[test]
fn composite_predicate_string_families_are_not_transitive() {
    let names = vec!["a".into(), "b".into()];
    let plain = Value::String("Oslo".into());
    let wrapped = Value::String("[\"Oslo\"]".into());
    let nested = Value::String("[\"[\"Oslo\"]\"]".into());
    let graph = composite_indexed(
        &names,
        vec![
            vec![plain.clone(), Value::Int64(1)],
            vec![wrapped.clone(), Value::Float64(1.0)],
            vec![nested, Value::UniqueId(1)],
        ],
    );
    assert_eq!(
        slots(graph.lookup_by_composite_predicate("N", &names, &[plain, Value::Int64(1)])),
        vec![0, 1]
    );
    assert_eq!(
        slots(graph.lookup_by_composite_predicate("N", &names, &[wrapped, Value::Int64(1)])),
        vec![0, 1, 2]
    );
}

#[test]
fn composite_predicate_declines_unproved_domains_without_partial_hits() {
    let names = vec!["a".into(), "b".into()];
    let graph = composite_indexed(&names, vec![vec![Value::Int64(1), Value::Int64(2)]]);
    for value in [Value::Int64(1 << 53), Value::List(vec![Value::Null])] {
        assert!(graph
            .lookup_by_composite_predicate("N", &names, &[Value::Int64(1), value])
            .is_none());
    }
    for value in [Value::Null, Value::Float64(f64::NAN)] {
        assert!(
            slots(graph.lookup_by_composite_predicate("N", &names, &[Value::Int64(1), value]))
                .is_empty()
        );
    }
    assert!(graph
        .lookup_by_composite_predicate("Absent", &names, &[Value::Null, Value::Int64(2)])
        .is_none());
    assert!(graph
        .lookup_by_composite_predicate("N", &names, &[Value::Int64(1)])
        .is_none());
}

#[test]
fn composite_predicate_probe_cap_is_all_or_nothing() {
    for (width, admitted) in [(6, true), (7, false)] {
        let names: Vec<_> = (0..width).map(|i| format!("p{i}")).collect();
        let values = vec![Value::String("plain".into()); width];
        let graph = composite_indexed(&names, vec![values.clone()]);
        let hits = graph.lookup_by_composite_predicate("N", &names, &values);
        if admitted {
            assert_eq!(slots(hits), vec![0]);
        } else {
            assert!(hits.is_none());
        }
    }
}

#[test]
fn composite_predicate_keeps_registered_aliases_and_declines_soft_aliases() {
    let names = vec!["rid".into(), "title_alias".into()];
    let values = vec![Value::Int64(1), Value::String("Oslo".into())];
    let mut graph = composite_indexed(&names, vec![values.clone()]);
    std::sync::Arc::make_mut(&mut graph.id_field_aliases).insert("N".into(), "rid".into());
    std::sync::Arc::make_mut(&mut graph.title_field_aliases)
        .insert("N".into(), "title_alias".into());
    assert_eq!(
        slots(graph.lookup_by_composite_predicate("N", &names, &values)),
        vec![0]
    );
    let names = vec!["name".into(), "v".into()];
    let values = vec![Value::String("Oslo".into()), Value::Boolean(true)];
    let graph = composite_indexed(&names, vec![values.clone()]);
    assert!(graph
        .lookup_by_composite_predicate("N", &names, &values)
        .is_none());
}
