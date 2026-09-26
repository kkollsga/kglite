use super::*;
use serde_json::json;

fn blueprint(nodes: serde_json::Value) -> Blueprint {
    serde_json::from_value(json!({ "nodes": nodes })).expect("fixture blueprint parses")
}

fn junction(temporal: serde_json::Value) -> serde_json::Value {
    json!({
        "Person": {
            "csv": "p.csv",
            "pk": "id",
            "connections": {"junction_edges": {"WORKS_AT": {
                "csv": "w.csv",
                "source_fk": "pid",
                "target": "Org",
                "target_fk": "oid",
                "properties": ["hired_on", "left_on"],
                "property_types": {"hired_on": "validFrom", "left_on": "validTo"},
                "rename": {"hired_on": "start", "left_on": "end"},
                "temporal": temporal
            }}}
        }
    })
}

#[test]
fn an_edge_bound_names_the_stored_property() {
    let ok = blueprint(junction(
        json!({"from": "start", "to": "end", "convention": "closed"}),
    ));
    assert_eq!(check_temporal_specs(&ok), Ok(Vec::new()));

    let csv_name = blueprint(junction(
        json!({"from": "hired_on", "to": "end", "convention": "closed"}),
    ));
    let err = check_temporal_specs(&csv_name).unwrap_err();
    assert!(err.contains("junction 'WORKS_AT' (node 'Person')"), "{err}");
    assert!(
        err.contains("'hired_on' is renamed to 'start'") && err.contains("'from': 'start'"),
        "{err}"
    );

    let unlisted = blueprint(junction(
        json!({"from": "start", "to": "gone", "convention": "closed"}),
    ));
    let err = check_temporal_specs(&unlisted).unwrap_err();
    assert!(err.contains("The edge stores: 'start', 'end'."), "{err}");
}

#[test]
fn an_fk_edge_bound_resolves_through_its_rename() {
    let bp = blueprint(json!({
        "Person": {
            "csv": "p.csv",
            "pk": "id",
            "connections": {"fk_edges": {"IN_ORG": {
                "target": "Org",
                "fk": "oid",
                "properties": ["a", "b"],
                "rename": {"a": "since"},
                "temporal": {"from": "since", "to": "b", "convention": "half_open"}
            }}}
        }
    }));
    assert_eq!(check_temporal_specs(&bp), Ok(Vec::new()));
}

#[test]
fn a_key_without_a_convention_warns() {
    let bp = blueprint(junction(json!({"from": "start", "to": "end"})));
    let warnings = check_temporal_specs(&bp).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains("names no convention") && warnings[0].contains("\"half_open\""),
        "{warnings:?}"
    );
}

#[test]
fn an_unknown_convention_is_refused() {
    let bp = blueprint(junction(
        json!({"from": "start", "to": "end", "convention": "half-open"}),
    ));
    let err = check_temporal_specs(&bp).unwrap_err();
    assert!(err.contains("convention 'half-open'"), "{err}");
}

#[test]
fn one_property_cannot_bound_both_ends() {
    let bp = blueprint(json!({
        "S": {"csv": "s.csv", "pk": "id",
              "temporal": {"from": "d", "to": "d", "convention": "closed"}}
    }));
    assert!(check_temporal_specs(&bp)
        .unwrap_err()
        .contains("both name 'd'"));
}

#[test]
fn role_types_alone_warn_with_the_stored_names() {
    let mut nodes = junction(json!(null));
    nodes["Person"]["connections"]["junction_edges"]["WORKS_AT"]
        .as_object_mut()
        .unwrap()
        .remove("temporal");
    nodes["Status"] = json!({
        "csv": "s.csv", "pk": "id",
        "properties": {"sf": "validFrom", "st": "validTo"}
    });
    let warnings = check_temporal_specs(&blueprint(nodes)).unwrap();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings.iter().any(|w| w.starts_with("junction 'WORKS_AT'")
            && w.contains("\"from\": \"start\", \"to\": \"end\"")),
        "{warnings:?}"
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.starts_with("node 'Status'") && w.contains("\"from\": \"sf\"")),
        "{warnings:?}"
    );
}

#[test]
fn a_sub_node_spec_is_checked_too() {
    let bp = blueprint(json!({
        "P": {"csv": "p.csv", "pk": "id", "sub_nodes": {
            "S": {"csv": "s.csv", "pk": "id",
                  "temporal": {"from": "a", "to": "b", "convention": "sideways"}}
        }}
    }));
    assert!(check_temporal_specs(&bp)
        .unwrap_err()
        .starts_with("node 'S'"));
}

#[test]
fn a_stray_key_inside_temporal_fails_the_parse() {
    let parsed: Result<Blueprint, _> = serde_json::from_value(json!({"nodes": {
        "S": {"csv": "s.csv", "temporal": {"from": "a", "to": "b", "conventon": "closed"}}
    }}));
    assert!(parsed.unwrap_err().to_string().contains("conventon"));
}
