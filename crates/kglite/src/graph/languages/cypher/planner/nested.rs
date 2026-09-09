//! Planning for nested queries and their imported outer scope.

use super::*;
use crate::graph::core::pattern_matching::PatternElement;

/// **Pass:** `optimize_nested_queries` — Recurse into UNION right-arms and
/// `CALL { }` bodies, inheriting diagnostic pass disables. Imported pattern
/// anchors disable graph-global fusions that cannot consume a per-row seed;
/// legacy set arms also retain their importing WITH boundaries.
pub(super) fn pass_optimize_nested_queries(query: &mut CypherQuery, ctx: &PassCtx) {
    let mut visible = ctx.initial_scope.clone();
    for clause in &mut query.clauses {
        match clause {
            Clause::Union(ref mut set) => optimize_with_disabled_scoped(
                &mut set.query,
                ctx.graph,
                ctx.params,
                ctx.disabled,
                ctx.initial_scope,
                ctx.global_scope,
            ),
            Clause::CallSubquery { import, body } => {
                optimize_call_body(import, body, &visible, ctx);
            }
            _ => {}
        }
        simplification::advance_visible_variable_scope(&mut visible, clause);
        visible.extend(ctx.global_scope.iter().cloned());
    }
}

fn optimize_call_body(
    import: &CallSubqueryImport,
    body: &mut CypherQuery,
    visible: &HashSet<String>,
    ctx: &PassCtx<'_>,
) {
    let imports: HashSet<String> = match import {
        CallSubqueryImport::Legacy(names) | CallSubqueryImport::Named(names) => {
            names.iter().cloned().collect()
        }
        CallSubqueryImport::All => visible.clone(),
        CallSubqueryImport::Empty => HashSet::new(),
    };
    let import_names: Vec<String> = imports.iter().cloned().collect();
    let anchors = import_pattern_anchors(body, &import_names);
    let empty_globals = HashSet::new();
    let body_globals = if matches!(
        import,
        CallSubqueryImport::Named(_) | CallSubqueryImport::All
    ) {
        &imports
    } else {
        &empty_globals
    };
    let legacy_set = matches!(import, CallSubqueryImport::Legacy(_))
        && body
            .clauses
            .iter()
            .any(|clause| matches!(clause, Clause::Union(_)));
    let mut disabled = ctx.disabled.clone();
    if legacy_set {
        disabled.insert("fold_pass_through_with".to_string());
    }
    if !anchors.is_empty() {
        disabled.extend(seed_ignoring_fusion_passes().iter().cloned());
    }
    optimize_with_disabled_scoped(
        body,
        ctx.graph,
        ctx.params,
        &disabled,
        &imports,
        body_globals,
    );
}

/// Imported names used as MATCH anchors anywhere in this subquery/set tree.
fn import_pattern_anchors(body: &CypherQuery, import: &[String]) -> Vec<String> {
    let mut anchors = Vec::new();
    collect_import_pattern_anchors(body, import, true, &mut anchors);
    anchors
}

/// Arm-local form used by seeded set execution. It stops at this arm's set
/// operator because an import can be scalar in one arm and an anchor in another.
pub(crate) fn import_pattern_anchors_in_arm(body: &CypherQuery, import: &[String]) -> Vec<String> {
    let mut anchors = Vec::new();
    collect_import_pattern_anchors(body, import, false, &mut anchors);
    anchors
}

fn collect_import_pattern_anchors(
    body: &CypherQuery,
    import: &[String],
    recurse_sets: bool,
    anchors: &mut Vec<String>,
) {
    for clause in &body.clauses {
        let patterns = match clause {
            Clause::Match(matched) | Clause::OptionalMatch(matched) => &matched.patterns,
            Clause::Union(set) if recurse_sets => {
                collect_import_pattern_anchors(&set.query, import, true, anchors);
                continue;
            }
            _ => continue,
        };
        for pattern in patterns {
            for element in &pattern.elements {
                let variable = match element {
                    PatternElement::Node(node) => node.variable.as_ref(),
                    PatternElement::Edge(edge) => edge.variable.as_ref(),
                };
                if let Some(variable) = variable {
                    if import.iter().any(|name| name == variable)
                        && !anchors.iter().any(|anchor| anchor == variable)
                    {
                        anchors.push(variable.clone());
                    }
                }
            }
        }
    }
}

/// Optimizer passes that emit graph-global operators and cannot consume a
/// per-row CALL-subquery seed. Names must remain registered in `PASSES`.
fn seed_ignoring_fusion_passes() -> &'static HashSet<String> {
    static PASSES_SET: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    PASSES_SET.get_or_init(|| {
        [
            "fuse_anchored_edge_count",
            "fuse_count_short_circuits",
            "fuse_optional_match_aggregate",
            "fuse_match_return_aggregate",
            "fuse_match_with_aggregate",
            "fuse_match_with_aggregate_top_k",
            "fuse_node_scan_aggregate",
            "fuse_node_scan_top_k",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    })
}
