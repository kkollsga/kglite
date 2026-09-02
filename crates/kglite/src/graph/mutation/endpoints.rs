//! Resolving a connection frame's endpoint ids to node indices.
//!
//! Split out of `maintain.rs`: the whole cluster is one job — walk an
//! `add_connections` frame once, decide per row whether both endpoints exist,
//! and hand back the matched rows, the rows held back for stub vivification,
//! and the null-id counts the caller reports.

use crate::datatypes::{DataFrame, Value};
use crate::graph::schema::DirGraph;
use crate::graph::storage::lookups::CombinedTypeLookup;
use petgraph::graph::NodeIndex;
use std::collections::HashSet;

/// What one `add_connections` frame's rows resolved to, produced before any
/// mutation runs (see the call site for why the split exists).
pub(super) struct ResolvedEndpoints {
    /// `(row, source, target)` for the rows whose endpoints both exist.
    pub(super) matched: Vec<(usize, NodeIndex, NodeIndex)>,
    /// Rows held back because an endpoint is missing, with their raw ids.
    /// The missing ids are vivified as provisional stubs (Pass B) and these
    /// rows replayed (Pass C) — an edge to a missing endpoint is never
    /// dropped.
    pub(super) deferred: Vec<(usize, Value, Value)>,
    /// Deduped, order-preserving ids to vivify.
    pub(super) missing_sources: Vec<Value>,
    pub(super) missing_targets: Vec<Value>,
    pub(super) null_source_rows: usize,
    pub(super) null_target_rows: usize,
}

/// Resolve every row's endpoints against the endpoint types' id indices.
///
/// Probes `graph.id_indices` in place when both types are overlay-resident —
/// the case for every heap-resident graph — and otherwise falls back to
/// exactly the lookup this replaced (`CombinedTypeLookup::from_id_indices`,
/// which materializes a base entry or, failing that, scans the graph).
pub(super) fn resolve_endpoints(
    graph: &DirGraph,
    df_data: &DataFrame,
    source_type: &str,
    target_type: &str,
    source_id_idx: usize,
    target_id_idx: usize,
) -> Result<ResolvedEndpoints, String> {
    if let Some(resolved) =
        graph
            .id_indices
            .with_overlay_type_pair(source_type, target_type, |source, target| {
                scan_endpoint_rows(
                    df_data,
                    source_id_idx,
                    target_id_idx,
                    |id| source.get(id),
                    |id| target.get(id),
                )
            })
    {
        return Ok(resolved);
    }
    let lookup = CombinedTypeLookup::from_id_indices(
        &graph.id_indices,
        &graph.graph,
        source_type.to_string(),
        target_type.to_string(),
    )?;
    Ok(scan_endpoint_rows(
        df_data,
        source_id_idx,
        target_id_idx,
        |id| lookup.check_source(id),
        |id| lookup.check_target(id),
    ))
}

/// The row walk both resolution paths share; generic over the two probes so
/// each one monomorphizes into a direct call.
fn scan_endpoint_rows(
    df_data: &DataFrame,
    source_id_idx: usize,
    target_id_idx: usize,
    check_source: impl Fn(&Value) -> Option<NodeIndex>,
    check_target: impl Fn(&Value) -> Option<NodeIndex>,
) -> ResolvedEndpoints {
    let mut out = ResolvedEndpoints {
        matched: Vec::with_capacity(df_data.row_count()),
        deferred: Vec::new(),
        missing_sources: Vec::new(),
        missing_targets: Vec::new(),
        null_source_rows: 0,
        null_target_rows: 0,
    };
    let mut seen_missing_source: HashSet<Value> = HashSet::new();
    let mut seen_missing_target: HashSet<Value> = HashSet::new();

    for row_idx in 0..df_data.row_count() {
        let source_id = match df_data.get_value_by_index(row_idx, source_id_idx) {
            Some(Value::Null) | None => {
                out.null_source_rows += 1;
                continue;
            }
            Some(id) => id,
        };
        let target_id = match df_data.get_value_by_index(row_idx, target_id_idx) {
            Some(Value::Null) | None => {
                out.null_target_rows += 1;
                continue;
            }
            Some(id) => id,
        };
        match (check_source(&source_id), check_target(&target_id)) {
            (Some(source_idx), Some(target_idx)) => {
                out.matched.push((row_idx, source_idx, target_idx))
            }
            (s_opt, t_opt) => {
                if s_opt.is_none() && seen_missing_source.insert(source_id.clone()) {
                    out.missing_sources.push(source_id.clone());
                }
                if t_opt.is_none() && seen_missing_target.insert(target_id.clone()) {
                    out.missing_targets.push(target_id.clone());
                }
                out.deferred.push((row_idx, source_id, target_id));
            }
        }
    }
    out
}

/// Resolve an already-extracted `(row, source_id, target_id)` list — the
/// deferred-row replay. Same index precedence as [`resolve_endpoints`];
/// returns one entry per input row, in order.
pub(super) fn resolve_pairs(
    graph: &DirGraph,
    source_type: &str,
    target_type: &str,
    rows: &[(usize, Value, Value)],
) -> Result<Vec<Option<(NodeIndex, NodeIndex)>>, String> {
    fn walk(
        rows: &[(usize, Value, Value)],
        check_source: impl Fn(&Value) -> Option<NodeIndex>,
        check_target: impl Fn(&Value) -> Option<NodeIndex>,
    ) -> Vec<Option<(NodeIndex, NodeIndex)>> {
        rows.iter()
            .map(|(_, source_id, target_id)| {
                match (check_source(source_id), check_target(target_id)) {
                    (Some(s), Some(t)) => Some((s, t)),
                    _ => None,
                }
            })
            .collect()
    }

    if let Some(resolved) =
        graph
            .id_indices
            .with_overlay_type_pair(source_type, target_type, |source, target| {
                walk(rows, |id| source.get(id), |id| target.get(id))
            })
    {
        return Ok(resolved);
    }
    let lookup = CombinedTypeLookup::from_id_indices(
        &graph.id_indices,
        &graph.graph,
        source_type.to_string(),
        target_type.to_string(),
    )?;
    Ok(walk(
        rows,
        |id| lookup.check_source(id),
        |id| lookup.check_target(id),
    ))
}

/// Column indices for the optional endpoint title fields, `None` for a field
/// that was not named or is not in the frame.
pub(super) fn title_column_indices(
    df_data: &DataFrame,
    source_title_field: Option<&str>,
    target_title_field: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    (
        source_title_field.and_then(|field| df_data.get_column_index(field)),
        target_title_field.and_then(|field| df_data.get_column_index(field)),
    )
}

/// Append a line per endpoint field whose null ids cost rows.
///
/// Genuine skips only: a row whose endpoint is *missing* is vivified as a stub
/// and replayed, not skipped, so it is never reported here.
pub(super) fn report_null_id_skips(
    errors: &mut Vec<String>,
    source: (usize, &str),
    target: (usize, &str),
) {
    for (skipped, field, side) in [
        (source.0, source.1, "source"),
        (target.0, target.1, "target"),
    ] {
        if skipped > 0 {
            errors.push(format!(
                "Skipped {skipped} rows: null values in {side} ID field '{field}'"
            ));
        }
    }
}
