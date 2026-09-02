use super::*;
use serde_json::json;

/// Every scalar, map and vec field set to a distinguishable value, so a clone
/// that drops one shows up as a `Debug` difference rather than as a silently
/// empty default.
fn fully_populated_spec() -> NodeSpec {
    serde_json::from_value(json!({
        "csv": "sample.csv",
        "pk": "sample_id",
        "title": "sample_name",
        "parent": "Study",
        "parent_fk": "study_id",
        "properties": {"depth": "float", "collected": "date"},
        "skipped": ["internal_note"],
        "filter": {"kingdom": "Bacteria"},
        "connections": {
            "fk_edges": {"IN_STUDY": {"target": "Study", "fk": "study_id"}},
            "junction_edges": {
                "HAS_TAXON": {
                    "csv": "sample_taxon.csv",
                    "source_fk": "sample_id",
                    "target": "Taxon",
                    "target_fk": "taxon_id",
                    "properties": ["abundance"],
                    "property_types": {"abundance": "float"},
                    "rename": {"abundance": "relative_abundance"}
                }
            }
        },
        "sub_nodes": {
            "Aliquot": {"pk": "aliquot_id", "parent_fk": "sample_id"}
        },
        "timeseries": {
            "time_key": {"year": "int", "month": "int"},
            "channels": {"ph": "float"},
            "resolution": "monthly",
            "units": {"ph": "pH"}
        }
    }))
    .expect("fixture spec parses")
}

/// The flattening pass copies a node spec minus its `sub_nodes`. A field the
/// copy forgets is a load-time directive silently lost for that type — the
/// blueprint declares it, the build ignores it, and nothing complains.
#[test]
fn clone_without_subs_keeps_every_field_except_sub_nodes() {
    let spec = fully_populated_spec();
    let mut expected = fully_populated_spec();
    expected.sub_nodes = IndexMap::new();

    let cloned = clone_without_subs(&spec);

    assert_eq!(format!("{cloned:?}"), format!("{expected:?}"));
    assert!(!spec.sub_nodes.is_empty(), "fixture must have a sub_node");
}
