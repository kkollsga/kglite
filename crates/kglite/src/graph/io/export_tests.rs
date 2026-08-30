//! Export regression tests.
//!
//! The columnar cases here are the red proof for a fixed export bug:
//! `property_iter` yields nothing for `PropertyStorage::Columnar`, so GraphML
//! and D3-JSON export silently emitted **empty** property sets for every node
//! of a saved (columnar) graph, while `property_count()` reported the real
//! count. Each test below asserts the same property survives export in *both*
//! the row-storage and the columnar shape, so the columnar arm fails on the
//! pre-fix code and the row arm proves the assertion is not vacuous.

use super::*;
use crate::datatypes::{DataFrame, Value};
use crate::graph::dir_graph::DirGraph;

/// GraphML embeds the property JSON as an XML-escaped attribute payload.
const GRAPHML_TOPIC_ALPHA: &str = "&quot;topic&quot;:&quot;alpha&quot;";

/// Two `Doc` nodes carrying a non-identity property each.
fn docs_graph() -> DirGraph {
    let mut g = DirGraph::new();
    let rows: Vec<Vec<Value>> = vec![
        vec![
            Value::Int64(1),
            Value::String("t1".into()),
            Value::String("alpha".into()),
            Value::Int64(10),
        ],
        vec![
            Value::Int64(2),
            Value::String("t2".into()),
            Value::String("beta".into()),
            Value::Int64(20),
        ],
    ];
    let df = DataFrame::from_cypher_rows(
        vec![
            "id".to_string(),
            "title".to_string(),
            "topic".to_string(),
            "score".to_string(),
        ],
        rows,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut g,
        df,
        "Doc".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    g
}

/// The same graph after `enable_columnar()` — the shape every `save()` /
/// reload produces, and the one the defect lived in.
fn columnar_docs_graph() -> DirGraph {
    let mut g = docs_graph();
    g.enable_columnar();
    // Precondition: the nodes really are columnar, otherwise the columnar
    // assertions below would be a restatement of the row-storage ones.
    let idx = g.graph.node_indices().next().unwrap();
    assert!(
        g.node_view(idx).unwrap().properties_are_columnar(),
        "enable_columnar() did not produce columnar nodes; test is vacuous"
    );
    g
}

#[test]
fn graphml_export_carries_properties_from_row_storage() {
    let g = docs_graph();
    let xml = to_graphml(&g, None).unwrap();
    assert!(xml.contains(GRAPHML_TOPIC_ALPHA), "got: {xml}");
    assert!(xml.contains("&quot;score&quot;:10"), "got: {xml}");
}

#[test]
fn graphml_export_carries_properties_from_columnar_storage() {
    let g = columnar_docs_graph();
    let xml = to_graphml(&g, None).unwrap();
    assert!(
        xml.contains(GRAPHML_TOPIC_ALPHA),
        "columnar GraphML export lost node properties; got: {xml}"
    );
    assert!(xml.contains("&quot;score&quot;:10"), "got: {xml}");
    assert!(
        !xml.contains("<data key=\"node_properties\">{}</data>"),
        "columnar GraphML export emitted an empty property object; got: {xml}"
    );
}

#[test]
fn d3_json_export_carries_properties_from_row_storage() {
    let g = docs_graph();
    let json = to_d3_json(&g, None).unwrap();
    assert!(json.contains("\"topic\":\"alpha\""), "got: {json}");
    assert!(json.contains("\"score\":10"), "got: {json}");
}

#[test]
fn d3_json_export_carries_properties_from_columnar_storage() {
    let g = columnar_docs_graph();
    let json = to_d3_json(&g, None).unwrap();
    assert!(
        json.contains("\"topic\":\"alpha\""),
        "columnar D3-JSON export lost node properties; got: {json}"
    );
    assert!(json.contains("\"score\":10"), "got: {json}");
}

/// The per-type CSV tree already read the columnar row correctly
/// (`property_keys` + `get_property`, both of which handle `Columnar`). Pinned
/// so the accessor migration cannot regress it into the `property_iter` shape.
#[test]
fn csv_dir_export_carries_properties_from_columnar_storage() {
    let g = columnar_docs_graph();
    let dir = tempfile::tempdir().unwrap();
    to_csv_dir(
        &g,
        dir.path().to_str().unwrap(),
        None,
        &std::collections::HashMap::new(),
    )
    .unwrap();
    let csv = std::fs::read_to_string(dir.path().join("nodes").join("Doc.csv")).unwrap();
    assert!(csv.contains("topic"), "got: {csv}");
    assert!(csv.contains("alpha"), "got: {csv}");
}

/// Gephi, yEd and Cytoscape all read `attr.name="label"` as a node's display
/// name. Without that key an import shows the synthetic `n0`/`n1` element ids
/// while the readable name sits under the non-standard `title` key, which no
/// reader looks at. GEXF's writer already gets this right (`label=` on the
/// node element), so GraphML was the odd one out.
#[test]
fn graphml_export_carries_a_label_key_from_the_node_title() {
    let g = docs_graph();
    let xml = to_graphml(&g, None).unwrap();
    assert!(
        xml.contains(
            "<key id=\"node_label\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>"
        ),
        "GraphML declared no node label key; got: {xml}"
    );
    assert!(
        xml.contains("<data key=\"node_label\">t1</data>"),
        "GraphML label key carried no title; got: {xml}"
    );
    // The pre-existing keys stay: readers already consuming them must not break.
    assert!(xml.contains("attr.name=\"title\""), "got: {xml}");
    assert!(xml.contains("attr.name=\"id\""), "got: {xml}");
}

/// Same story on the edge side: `attr.name="label"` is what a reader renders
/// on the arc, and the connection type is the only label kglite has.
#[test]
fn graphml_export_carries_an_edge_label_key_from_the_connection_type() {
    let mut g = docs_graph();
    crate::graph::mutation::maintain::add_edges_from_specs(
        &mut g,
        vec![crate::graph::mutation::maintain::EdgeSpec {
            source_type: "Doc".to_string(),
            source_id: Value::Int64(1),
            target_type: "Doc".to_string(),
            target_id: Value::Int64(2),
            edge_type: "CITES".to_string(),
            properties: std::collections::HashMap::new(),
        }],
    )
    .unwrap();
    let xml = to_graphml(&g, None).unwrap();
    assert!(
        xml.contains(
            "<key id=\"edge_label\" for=\"edge\" attr.name=\"label\" attr.type=\"string\"/>"
        ),
        "GraphML declared no edge label key; got: {xml}"
    );
    assert!(
        xml.contains("<data key=\"edge_label\">CITES</data>"),
        "GraphML edge label key carried no connection type; got: {xml}"
    );
    assert!(xml.contains("attr.name=\"connection_type\""), "got: {xml}");
}
