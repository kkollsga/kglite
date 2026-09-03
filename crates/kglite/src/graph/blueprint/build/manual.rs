//! Phase 1: manual nodes — node types with no input of their own, synthesised
//! from the FK values of the specs that refer to them.

use super::super::input::InputRegistry;
use super::fk::{add_id_column, infer_id_type};
use super::specs::FlatSpec;
use super::BuildReport;
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::DirGraph;
use std::collections::HashSet;

pub(super) fn load_manual_nodes(
    graph: &mut DirGraph,
    core: &[FlatSpec],
    subs: &[FlatSpec],
    registry: &InputRegistry,
    report: &mut BuildReport,
) -> Result<(), String> {
    let manual: Vec<&FlatSpec> = core.iter().filter(|s| s.is_manual).collect();
    if manual.is_empty() {
        return Ok(());
    }

    for ms in &manual {
        let mut distinct: HashSet<String> = HashSet::new();
        // Scan every spec's fk_edges for targets pointing at this manual type.
        for spec in core.iter().chain(subs.iter()) {
            let Some(input) = spec.input.as_deref() else {
                continue;
            };
            for (_, edge) in &spec.spec.connections.fk_edges {
                if edge.target != ms.node_type {
                    continue;
                }
                let raw = match registry.get(input).and_then(|s| s.read_all()) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                if let Some(fk_idx) = raw.col_index(&edge.fk) {
                    for (r, row) in raw.rows.iter().enumerate() {
                        if raw.nulls[r][fk_idx] {
                            continue;
                        }
                        let trimmed = row[fk_idx].trim();
                        if !trimmed.is_empty() {
                            distinct.insert(trimmed.to_string());
                        }
                    }
                }
            }
        }

        if distinct.is_empty() {
            continue;
        }

        let pk = ms.spec.pk.clone().unwrap_or_else(|| "name".to_string());
        let title = ms.spec.title.clone().unwrap_or_else(|| pk.clone());

        let mut df = DataFrame::new(Vec::new());
        let values: Vec<Option<String>> = distinct.into_iter().map(Some).collect();
        // The id column is typed by exactly the rule the FK-edge frame uses
        // (`infer_id_type`), because these two frames must agree: the edge
        // resolves its endpoint against the ids this call creates. Typed
        // `String` unconditionally, a numeric FK produced a `"10"` node here
        // and an unmatched `10` endpoint there — so the edge vivified a second
        // node and the type carried two per value.
        let id_type = infer_id_type(&values);
        add_id_column(&mut df, &pk, values.clone(), id_type)
            .map_err(|e| format!("manual nodes: {}", e))?;
        if title != pk {
            // The title is a display string whatever the id is.
            let data2 = crate::datatypes::values::ColumnData::String(values);
            df.add_column(
                title.clone(),
                crate::datatypes::values::ColumnType::String,
                data2,
            )
            .map_err(|e| format!("manual nodes: {}", e))?;
        }

        let title_field = if title != pk {
            Some(title.clone())
        } else {
            None
        };
        let result = maintain::add_nodes(graph, df, ms.node_type.clone(), pk, title_field, None)
            .map_err(|e| format!("manual nodes '{}': {}", ms.node_type, e))?;

        let count = result.nodes_created + result.nodes_updated;
        report.nodes_by_type.insert(ms.node_type.clone(), count);
    }

    Ok(())
}
