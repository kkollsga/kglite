//! Phase 1: manual nodes — node types with no input of their own, synthesised
//! from the FK values of the specs that refer to them.

use super::super::input::InputRegistry;
use super::super::table::RawCsv;
use super::cache::CsvCache;
use super::fk::{add_id_column, infer_id_type};
use super::nodes::{node_chunk_size, should_stream_spec};
use super::specs::FlatSpec;
use super::BuildReport;
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::DirGraph;
use std::collections::{HashMap, HashSet};

pub(super) fn load_manual_nodes(
    graph: &mut DirGraph,
    core: &[FlatSpec],
    subs: &[FlatSpec],
    registry: &InputRegistry,
    cache: &CsvCache,
    report: &mut BuildReport,
) -> Result<(), String> {
    let manual: Vec<&FlatSpec> = core.iter().filter(|s| s.is_manual).collect();
    if manual.is_empty() {
        return Ok(());
    }
    let manual_types: HashSet<&str> = manual.iter().map(|m| m.node_type.as_str()).collect();

    let distinct = scan_fk_values(core, subs, &manual_types, registry, cache);

    for ms in &manual {
        let Some(values) = distinct.get(ms.node_type.as_str()) else {
            continue;
        };
        if values.is_empty() {
            continue;
        }

        let pk = ms.spec.pk.clone().unwrap_or_else(|| "name".to_string());
        let title = ms.spec.title.clone().unwrap_or_else(|| pk.clone());

        let mut df = DataFrame::new(Vec::new());
        let values: Vec<Option<String>> = values.iter().cloned().map(Some).collect();
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

/// The distinct FK values each manual type is synthesised from, gathered in
/// **one pass per input**.
///
/// Every referring spec is read once and every manual-target FK column of that
/// spec is harvested from the same table. Reading per (manual type × edge)
/// instead — as this phase used to — costs M×K whole-file reads of the same
/// input before the node phases have cached anything.
fn scan_fk_values<'a>(
    core: &'a [FlatSpec],
    subs: &'a [FlatSpec],
    manual_types: &HashSet<&str>,
    registry: &InputRegistry,
    cache: &CsvCache,
) -> HashMap<&'a str, HashSet<String>> {
    let mut distinct: HashMap<&str, HashSet<String>> = HashMap::new();
    for spec in core.iter().chain(subs.iter()) {
        let Some(input) = spec.input.as_deref() else {
            continue;
        };
        // (FK column, manual type) for every edge of this spec that feeds a
        // manual type; a spec with none is never read.
        let wanted: Vec<(&str, &str)> = spec
            .spec
            .connections
            .fk_edges
            .values()
            .filter(|e| manual_types.contains(e.target.as_str()))
            .map(|e| (e.fk.as_str(), e.target.as_str()))
            .collect();
        if wanted.is_empty() {
            continue;
        }

        // Held per spec and merged only on a clean read, so an input that
        // fails part-way contributes nothing — the same all-or-nothing the
        // whole-file read gave.
        let mut found: HashMap<&str, HashSet<String>> = HashMap::new();
        let ok = if should_stream_spec(spec, registry) {
            // A spec big enough to stream is never materialised here either:
            // the manual phase runs before the node phases, so a whole-file
            // read would hold the peak the streaming path exists to avoid.
            match registry
                .get(input)
                .and_then(|s| s.chunks(node_chunk_size()))
            {
                Ok(chunks) => chunks.into_iter().try_fold((), |_, chunk| {
                    chunk.map(|raw| harvest(&raw, &wanted, &mut found))
                }),
                Err(e) => Err(e),
            }
        } else {
            cache
                .get(registry, input)
                .map(|raw| harvest(&raw, &wanted, &mut found))
        };
        if ok.is_err() {
            // Unreadable inputs are non-fatal here: the node phase reports the
            // error against the spec that owns the file.
            continue;
        }
        for (target, values) in found {
            distinct.entry(target).or_default().extend(values);
        }
    }
    distinct
}

/// Add one table's non-empty values for each wanted (column, manual type).
fn harvest<'a>(
    raw: &RawCsv,
    wanted: &[(&str, &'a str)],
    found: &mut HashMap<&'a str, HashSet<String>>,
) {
    for (fk, target) in wanted {
        let Some(fk_idx) = raw.col_index(fk) else {
            continue;
        };
        let bucket = found.entry(target).or_default();
        for (r, row) in raw.rows.iter().enumerate() {
            if raw.nulls[r][fk_idx] {
                continue;
            }
            let trimmed = row[fk_idx].trim();
            if !trimmed.is_empty() {
                bucket.insert(trimmed.to_string());
            }
        }
    }
}

#[cfg(test)]
mod manual_node_tests {
    use super::super::super::input::csv::CsvFile;
    use super::super::super::input::test_double::CountingSource;
    use super::super::super::input::InputRegistry;
    use super::*;
    use std::io::Write;

    fn spec_from(json: serde_json::Value) -> super::super::super::schema::NodeSpec {
        serde_json::from_value(json).expect("fixture spec parses")
    }

    /// One CSV, two manual types, three FK edges pointing at them: the loader
    /// must open the input once, not once per (manual type × edge).
    #[test]
    fn one_input_is_opened_once_however_many_manual_edges_reference_it() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "id,owner,co_owner,tag").unwrap();
        writeln!(f, "1,ann,bob,red").unwrap();
        writeln!(f, "2,bob,ann,blue").unwrap();
        f.flush().unwrap();

        let (counting, opens) = CountingSource::new(Box::new(CsvFile::new(
            f.path().to_path_buf(),
            "items.csv".to_string(),
        )));
        let mut registry = InputRegistry::default();
        registry.insert("items.csv", Box::new(counting));

        let core = vec![
            FlatSpec {
                node_type: "Owner".to_string(),
                spec: spec_from(serde_json::json!({"pk": "name"})),
                parent: None,
                is_manual: true,
                input: None,
            },
            FlatSpec {
                node_type: "Tag".to_string(),
                spec: spec_from(serde_json::json!({"pk": "name"})),
                parent: None,
                is_manual: true,
                input: None,
            },
            FlatSpec {
                node_type: "Item".to_string(),
                spec: spec_from(serde_json::json!({
                    "csv": "items.csv",
                    "pk": "id",
                    "connections": {"fk_edges": {
                        "OWNED_BY": {"target": "Owner", "fk": "owner"},
                        "CO_OWNED_BY": {"target": "Owner", "fk": "co_owner"},
                        "TAGGED": {"target": "Tag", "fk": "tag"}
                    }}
                })),
                parent: None,
                is_manual: false,
                input: Some("items.csv".to_string()),
            },
        ];

        let mut graph = DirGraph::new();
        let mut report = BuildReport {
            nodes_by_type: Default::default(),
            edges_by_type: Default::default(),
            edges_actual: Default::default(),
            warnings: Vec::new(),
            errors: Vec::new(),
            provisional_purged: 0,
        };
        let cache = CsvCache::default();
        load_manual_nodes(&mut graph, &core, &[], &registry, &cache, &mut report).unwrap();

        assert_eq!(
            report.nodes_by_type.get("Owner").copied(),
            Some(2),
            "ann + bob across both owner columns"
        );
        assert_eq!(report.nodes_by_type.get("Tag").copied(), Some(2));
        assert_eq!(
            opens.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the input is read once for the whole manual phase"
        );
    }
}
