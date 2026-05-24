use super::*;
use crate::graph::languages::cypher::parser::parse_cypher;
use crate::graph::schema::{DirGraph, SpatialConfig};

fn graph_with_spatial() -> DirGraph {
    let mut g = DirGraph::new();
    g.spatial_configs.insert(
        "Area".into(),
        SpatialConfig {
            geometry: Some("geom".into()),
            location: None,
            points: Default::default(),
            shapes: Default::default(),
        },
    );
    g.spatial_configs.insert(
        "City".into(),
        SpatialConfig {
            geometry: None,
            location: Some(("lat".into(), "lon".into())),
            points: Default::default(),
            shapes: Default::default(),
        },
    );
    g
}

#[test]
fn rewrites_canonical_two_pattern_contains() {
    let mut q = parse_cypher("MATCH (a:Area), (c:City) WHERE contains(a, c) RETURN a.name, c.name")
        .unwrap();
    fuse_spatial_join(&mut q, &graph_with_spatial());
    assert!(matches!(&q.clauses[0], Clause::SpatialJoin { .. }));
    // WHERE consumed, RETURN remains
    assert_eq!(q.clauses.len(), 2);
    assert!(matches!(&q.clauses[1], Clause::Return(_)));
}

#[test]
fn rewrites_with_and_remainder() {
    let mut q = parse_cypher(
        "MATCH (a:Area), (c:City) WHERE contains(a, c) AND c.name = 'x' RETURN a.name",
    )
    .unwrap();
    fuse_spatial_join(&mut q, &graph_with_spatial());
    if let Clause::SpatialJoin { remainder, .. } = &q.clauses[0] {
        assert!(remainder.is_some(), "remainder should carry c.name = 'x'");
    } else {
        panic!("expected SpatialJoin, got {:?}", q.clauses[0]);
    }
}

#[test]
fn skips_constant_point() {
    let mut q =
        parse_cypher("MATCH (a:Area), (c:City) WHERE contains(a, point(60.0, 10.0)) RETURN a")
            .unwrap();
    fuse_spatial_join(&mut q, &graph_with_spatial());
    assert!(
        !q.clauses
            .iter()
            .any(|c| matches!(c, Clause::SpatialJoin { .. })),
        "constant-point case must not rewrite"
    );
}

#[test]
fn skips_negated_contains() {
    let mut q = parse_cypher("MATCH (a:Area), (c:City) WHERE NOT contains(a, c) RETURN a").unwrap();
    fuse_spatial_join(&mut q, &graph_with_spatial());
    assert!(
        !q.clauses
            .iter()
            .any(|c| matches!(c, Clause::SpatialJoin { .. })),
        "NOT contains must fall back to existing path"
    );
}

#[test]
fn skips_three_patterns() {
    let mut q =
        parse_cypher("MATCH (a:Area), (b:Area), (c:City) WHERE contains(a, c) RETURN a").unwrap();
    fuse_spatial_join(&mut q, &graph_with_spatial());
    assert!(
        !q.clauses
            .iter()
            .any(|c| matches!(c, Clause::SpatialJoin { .. })),
        "three-pattern MATCH must not rewrite"
    );
}

#[test]
fn skips_without_spatial_config() {
    let mut q = parse_cypher("MATCH (a:Foo), (c:Bar) WHERE contains(a, c) RETURN a").unwrap();
    fuse_spatial_join(&mut q, &DirGraph::new());
    assert!(
        !q.clauses
            .iter()
            .any(|c| matches!(c, Clause::SpatialJoin { .. })),
        "types without SpatialConfig must not rewrite"
    );
}

#[test]
fn skips_edge_pattern() {
    let mut q =
        parse_cypher("MATCH (a:Area)-[:R]->(x), (c:City) WHERE contains(a, c) RETURN a").unwrap();
    fuse_spatial_join(&mut q, &graph_with_spatial());
    assert!(
        !q.clauses
            .iter()
            .any(|c| matches!(c, Clause::SpatialJoin { .. })),
        "patterns with edges must not rewrite"
    );
}
