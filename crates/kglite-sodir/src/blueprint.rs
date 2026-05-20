//! Blueprint helpers — which datasets a blueprint references, and the
//! deep-merge of a base blueprint with an optional complement.
//!
//! Ported from the Python `wrapper.py` (`_datasets_used_by_blueprint`,
//! `_deep_merge`).

use std::collections::BTreeSet;

use serde_json::Value;

/// Walk a blueprint's node defs, junction edges and sub-nodes; return
/// the dataset stems (CSV filename without directory or `.csv`) it
/// references, sorted and de-duplicated.
pub fn datasets_used_by_blueprint(blueprint: &Value) -> Vec<String> {
    let mut stems: BTreeSet<String> = BTreeSet::new();
    if let Some(nodes) = blueprint.get("nodes").and_then(Value::as_object) {
        for node_def in nodes.values() {
            collect_csv(node_def.get("csv"), &mut stems);
            if let Some(junctions) = node_def
                .get("connections")
                .and_then(|c| c.get("junction_edges"))
                .and_then(Value::as_object)
            {
                for edge in junctions.values() {
                    collect_csv(edge.get("csv"), &mut stems);
                }
            }
            if let Some(subs) = node_def.get("sub_nodes").and_then(Value::as_object) {
                for sub in subs.values() {
                    collect_csv(sub.get("csv"), &mut stems);
                }
            }
        }
    }
    stems.into_iter().collect()
}

fn collect_csv(csv: Option<&Value>, out: &mut BTreeSet<String>) {
    if let Some(path) = csv.and_then(Value::as_str) {
        out.insert(stem_of(path));
    }
}

/// Filename without directory or trailing `.csv` — the Rust analogue
/// of Python's `Path(p).stem` for these blueprint CSV references.
fn stem_of(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    file.strip_suffix(".csv").unwrap_or(file).to_string()
}

/// Deep-merge `complement` onto `base`. Nested objects merge
/// recursively; arrays and scalars are replaced wholesale. On a leaf
/// collision the base value wins unless `complement_overrides` is set.
pub fn deep_merge(base: &Value, complement: &Value, complement_overrides: bool) -> Value {
    let (Value::Object(a), Value::Object(b)) = (base, complement) else {
        return base.clone();
    };
    let mut out = a.clone();
    for (k, v) in b {
        let recurse = v.is_object() && out.get(k).is_some_and(Value::is_object);
        if recurse {
            let merged = deep_merge(out.get(k).unwrap(), v, complement_overrides);
            out.insert(k.clone(), merged);
        } else if out.contains_key(k) && !complement_overrides {
            // Leaf collision — preserve the base value.
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn walks_nodes_junctions_and_subnodes() {
        let bp = json!({
            "nodes": {
                "Field": {
                    "csv": "field.csv",
                    "connections": {
                        "junction_edges": {
                            "HAS_OPERATOR": {"csv": "csv/field_operator_hst.csv"}
                        }
                    },
                    "sub_nodes": {
                        "Reserves": {"csv": "field_reserves.csv"}
                    }
                },
                "Wellbore": {"csv": "wellbore.csv"}
            }
        });
        let stems = datasets_used_by_blueprint(&bp);
        assert_eq!(
            stems,
            vec!["field", "field_operator_hst", "field_reserves", "wellbore"]
        );
    }

    #[test]
    fn deep_merge_base_wins_by_default() {
        let base = json!({"a": 1, "nested": {"x": "base"}});
        let comp = json!({"a": 99, "b": 2, "nested": {"x": "comp", "y": "new"}});
        let merged = deep_merge(&base, &comp, false);
        assert_eq!(merged["a"], json!(1)); // base wins on leaf collision
        assert_eq!(merged["b"], json!(2)); // new key added
        assert_eq!(merged["nested"]["x"], json!("base")); // recursive, base wins
        assert_eq!(merged["nested"]["y"], json!("new")); // recursive, new key
    }

    #[test]
    fn deep_merge_complement_overrides() {
        let base = json!({"a": 1});
        let comp = json!({"a": 99});
        let merged = deep_merge(&base, &comp, true);
        assert_eq!(merged["a"], json!(99));
    }
}
