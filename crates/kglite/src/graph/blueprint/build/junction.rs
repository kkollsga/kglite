//! Phase 5: junction edges — many-to-many CSVs with two FK columns and
//! optional property columns, always streamed in row chunks.

use super::super::input::InputRegistry;
use super::super::table::{ListMisparseTally, RawCsv};
use super::super::typing::{map_blueprint_type, typed_dataframe};
use super::cache::CsvCache;
use super::fk::connect;
use super::specs::FlatSpec;
use super::table_ops::subset_rows;
use super::BuildReport;
use crate::graph::mutation::maintain;
use crate::graph::schema::DirGraph;
use std::collections::{BTreeMap, HashMap};

// Streaming bounds peak RAM at chunk_size × cols × avg_string_len whatever
// the junction CSV's total size — the 10M+ row junction tables (e.g. SEC
// HOLDS at full-universe scale) are why.

/// Junction-edge chunk size. ~100K rows × ~10 columns × ~20B avg
/// string ≈ 20 MB peak per chunk, well under any reasonable RAM
/// budget for the build host. Configurable via env var for
/// performance experiments.
fn junction_chunk_size() -> usize {
    std::env::var("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
}

pub(super) fn load_junction_edges(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
    registry: &InputRegistry,
    _cache: &CsvCache,
    report: &mut BuildReport,
) -> Result<(), String> {
    let chunk_size = junction_chunk_size();
    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();
    let t_total = std::time::Instant::now();

    // Junction inputs are streamed, never cached: each is read exactly once,
    // so the `CsvCache` that node specs rely on would buy nothing here —
    // hence the unused `_cache` parameter.
    for spec in specs {
        for (edge_type, junc) in &spec.spec.connections.junction_edges {
            load_one_junction_edge(graph, spec, edge_type, junc, registry, chunk_size, report)?;
        }
    }

    if profile {
        eprintln!(
            "    streaming junction edges total: {} ms (chunk_size={})",
            t_total.elapsed().as_millis(),
            chunk_size,
        );
    }
    Ok(())
}

/// Stream one junction CSV into edges. One `connect` call per (chunk, target
/// type) — a single-target junction, which is every one written before the
/// union form, has exactly one group and takes the same path it always did.
fn load_one_junction_edge(
    graph: &mut DirGraph,
    spec: &FlatSpec,
    edge_type: &str,
    junc: &super::super::schema::JunctionEdge,
    registry: &InputRegistry,
    chunk_size: usize,
    report: &mut BuildReport,
) -> Result<(), String> {
    let mut keep: Vec<String> = vec![junc.source_fk.clone(), junc.target_fk.clone()];
    for p in &junc.properties {
        if !keep.contains(p) {
            keep.push(p.clone());
        }
    }
    let mut declared: HashMap<String, String> = HashMap::new();
    for (col, ty) in &junc.property_types {
        if map_blueprint_type(ty).is_some() {
            declared.insert(col.clone(), ty.clone());
        }
    }

    let rename = match junction_rename_map(edge_type, junc, &keep) {
        Ok(map) => map,
        Err(e) => {
            report.errors.push(e);
            return Ok(());
        }
    };
    let routing = match target_routing(edge_type, junc) {
        Ok(r) => r,
        Err(e) => {
            report.errors.push(e);
            return Ok(());
        }
    };

    // Decided before the first chunk and reused for all of them: streaming
    // this input bounds peak RAM, so it must not decide which of its rows become
    // parallel edges. Per-type splitting rides on the same decision — every
    // group of every chunk gets this one value. See `maintain::InitialLoad`.
    let initial_load =
        maintain::InitialLoad::Preset(!graph.connection_type_metadata.contains_key(edge_type));

    let chunks = match registry.get(&junc.csv).and_then(|s| s.chunks(chunk_size)) {
        Ok(it) => it,
        Err(e) => {
            report.errors.push(format!("junction {}: {}", edge_type, e));
            return Ok(());
        }
    };

    // One tally per junction input, not per chunk — see the node loader's for
    // why. Same for the unroutable-row counts.
    let mut misparses = ListMisparseTally::default();
    let mut unroutable: BTreeMap<String, usize> = BTreeMap::new();
    let mut reported_missing_type_column = false;

    for chunk_result in chunks {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                report.errors.push(format!("junction {}: {}", edge_type, e));
                continue;
            }
        };
        let chunk_keep: Vec<String> = keep
            .iter()
            .filter(|p| chunk.col_index(p).is_some())
            .cloned()
            .collect();
        if chunk_keep.is_empty() {
            continue;
        }
        if let TargetRouting::Column { column, .. } = &routing {
            if chunk.col_index(column).is_none() {
                if !reported_missing_type_column {
                    reported_missing_type_column = true;
                    report.errors.push(format!(
                        "junction {edge_type}: target_type_column '{column}' not found in the CSV"
                    ));
                }
                continue;
            }
        }

        for (target_type, rows) in
            group_rows_by_target(graph, &chunk, junc, &routing, &declared, &mut unroutable)
        {
            let whole_chunk = rows.len() == chunk.row_count();
            let subset;
            let source: &RawCsv = if whole_chunk {
                &chunk
            } else {
                subset = subset_rows(&chunk, &rows);
                &subset
            };
            let df = match typed_dataframe(source, &chunk_keep, &declared, &rename, &mut misparses)
            {
                Ok(df) => df,
                Err(e) => {
                    report.errors.push(format!("junction {}: {}", edge_type, e));
                    continue;
                }
            };
            let count = connect(
                graph,
                df,
                edge_type,
                &spec.node_type,
                &junc.source_fk,
                target_type,
                &junc.target_fk,
                report,
                initial_load,
            )?;
            *report
                .edges_by_type
                .entry(edge_type.to_string())
                .or_insert(0) += count;
        }
    }
    report.warnings.extend(misparses.into_warnings(&format!(
        "junction '{edge_type}' (node '{}')",
        spec.node_type
    )));
    for (value, count) in unroutable {
        report.warnings.push(format!(
            "junction '{edge_type}' (node '{}'): {count} row(s) name target type '{value}', \
             which is not in this edge's 'target' list ({}); they built no edge",
            spec.node_type,
            junc.target.join(", ")
        ));
    }
    Ok(())
}

/// How each row of a junction CSV picks its target node type.
enum TargetRouting<'a> {
    /// One declared type: every row goes to it.
    Single(&'a str),
    /// A CSV column names the type per row.
    Column {
        types: &'a [String],
        column: &'a str,
    },
    /// No column: the declared types are probed in declaration order and the
    /// first that already has the row's target id wins.
    Probe { types: &'a [String] },
}

fn target_routing<'a>(
    edge_type: &str,
    junc: &'a super::super::schema::JunctionEdge,
) -> Result<TargetRouting<'a>, String> {
    if junc.target.is_empty() {
        return Err(format!("junction {edge_type}: 'target' names no node type"));
    }
    let Some(column) = junc.target_type_column.as_deref() else {
        return Ok(if junc.target.len() == 1 {
            TargetRouting::Single(&junc.target[0])
        } else {
            TargetRouting::Probe {
                types: &junc.target,
            }
        });
    };
    if column == junc.source_fk || column == junc.target_fk {
        return Err(format!(
            "junction {edge_type}: target_type_column '{column}' is an fk column — it names the \
             column holding each row's target *type*, not its id"
        ));
    }
    Ok(TargetRouting::Column {
        types: &junc.target,
        column,
    })
}

/// One chunk's rows grouped by the target type they route to, in declaration
/// order, skipping types no row picked.
///
/// A row naming a type outside the declaration is counted in `unroutable` and
/// left out: the loader cannot guess which of the declared types the author
/// meant, and picking one would attach the edge to a node of the wrong type
/// — or invent a stub under it. Under the probe form there is no such row: an
/// id no declared type has takes the first declared type, where the existing
/// missing-endpoint policy vivifies its stub, because dropping it would lose
/// an edge the author did declare.
fn group_rows_by_target<'a>(
    graph: &DirGraph,
    chunk: &RawCsv,
    junc: &'a super::super::schema::JunctionEdge,
    routing: &TargetRouting<'a>,
    declared: &HashMap<String, String>,
    unroutable: &mut BTreeMap<String, usize>,
) -> Vec<(&'a str, Vec<usize>)> {
    let types: &[String] = match routing {
        TargetRouting::Single(t) => return vec![(t, (0..chunk.row_count()).collect())],
        TargetRouting::Column { types, .. } | TargetRouting::Probe { types } => types,
    };
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); types.len()];

    match routing {
        TargetRouting::Single(_) => unreachable!("returned above"),
        TargetRouting::Column { column, .. } => {
            let col_idx = chunk.col_index(column).expect("caller checked presence");
            for r in 0..chunk.row_count() {
                let value = &chunk.rows[r][col_idx];
                match types.iter().position(|t| t == value) {
                    Some(i) => groups[i].push(r),
                    None => *unroutable.entry(value.clone()).or_default() += 1,
                }
            }
        }
        TargetRouting::Probe { .. } => {
            let ids = typed_target_ids(chunk, &junc.target_fk, declared);
            for (r, id) in ids.iter().enumerate() {
                let owner = id.as_ref().and_then(|v| {
                    types
                        .iter()
                        .position(|t| graph.id_indices.lookup(t, v).is_some())
                });
                groups[owner.unwrap_or(0)].push(r);
            }
        }
    }

    types
        .iter()
        .zip(groups)
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(t, rows)| (t.as_str(), rows))
        .collect()
}

/// The chunk's target-id column, typed the way the edge frame will type it,
/// so a probe compares like with like: an id index keyed by `Int64` never
/// matches the raw `"10"` string the CSV holds.
fn typed_target_ids(
    chunk: &RawCsv,
    target_fk: &str,
    declared: &HashMap<String, String>,
) -> Vec<Option<crate::datatypes::values::Value>> {
    let keep = vec![target_fk.to_string()];
    // Discarded: the real frame types the same column again and tallies it
    // there, and counting a misparse twice would double the warning's count.
    let mut scratch = ListMisparseTally::default();
    match typed_dataframe(chunk, &keep, declared, &HashMap::new(), &mut scratch) {
        Ok(df) => (0..chunk.row_count())
            .map(|r| df.get_value_by_index(r, 0))
            .collect(),
        Err(_) => vec![None; chunk.row_count()],
    }
}

/// Validate a junction's `rename:` map and return it as the lookup
/// `typed_dataframe` takes. Keys must be property columns (fk columns keep
/// their spelling — `connect` finds them by name), and targets must not
/// collide with any kept column or another target.
fn junction_rename_map(
    edge_type: &str,
    junc: &crate::graph::blueprint::schema::JunctionEdge,
    keep: &[String],
) -> Result<HashMap<String, String>, String> {
    let mut rename: HashMap<String, String> = HashMap::new();
    for (col, new_name) in &junc.rename {
        if col == &junc.source_fk || col == &junc.target_fk {
            return Err(format!(
                "junction {edge_type}: rename of fk column '{col}' is not supported — \
                 source_fk/target_fk name the CSV columns"
            ));
        }
        if !junc.properties.contains(col) {
            return Err(format!(
                "junction {edge_type}: rename key '{col}' is not in 'properties'"
            ));
        }
        let collides = keep.iter().any(|k| k == new_name && k != col)
            || rename.values().any(|v| v == new_name);
        if collides {
            return Err(format!(
                "junction {edge_type}: rename target '{new_name}' collides with another column"
            ));
        }
        rename.insert(col.clone(), new_name.clone());
    }
    Ok(rename)
}
