//! Closure-probe eligibility — the single description of when
//! `MATCH (n:ClosedSupertype {p: v})` can be answered by per-member index
//! lookups instead of a scan.
//!
//! Two consumers read this module: the matcher, which runs the probe
//! ([`super::matcher::PatternExecutor::try_closure_probe`]), and
//! `generate_explain_result`, which renders the `ClosureProbe` plan row.
//! Keeping the enumeration and the coverage rule here is what stops the plan
//! and the execution drifting apart — EXPLAIN renders from
//! [`closure_probe_members`], and the matcher shares
//! [`live_closure_members`] with it.
//!
//! [`closure_probe_members`] is deliberately **value-independent**: it answers
//! "is every member's lookup servable by an index", never "does this value
//! exist". A value miss is a property of the data, not of the plan.

use crate::graph::schema::DirGraph;

/// The member types a closure probe on `node_type` would visit: every declared
/// descendant of `node_type` plus `node_type` itself, restricted to those with
/// at least one live node of that primary type. Declaration order is
/// `ontology.classes`' (a `BTreeMap`), so the sequence is deterministic, with
/// `node_type` last.
///
/// Liveness matters because a probe over a member with no nodes contributes
/// nothing but still demands index coverage the type may never have been given.
pub(crate) fn live_closure_members<'a>(graph: &'a DirGraph, node_type: &'a str) -> Vec<&'a str> {
    let mut members: Vec<&str> = graph
        .ontology
        .classes
        .iter()
        .filter(|(name, _)| {
            graph
                .ontology
                .ancestors(name)
                .iter()
                .any(|a| a == node_type)
        })
        .map(|(name, _)| name.as_str())
        .collect();
    members.push(node_type);
    members.retain(|member| {
        graph
            .type_indices
            .get(member)
            .is_some_and(|nodes| nodes.iter().next().is_some())
    });
    members
}

/// Whether a closure probe on `node_type` with equality predicates on
/// `props` is eligible, and over which member types it would run.
///
/// Eligible when *all* of:
/// - `node_type` is a **Closed** managed label — the engine is the bucket's
///   only writer, so the per-member probes are the complete answer. An `Open`
///   bucket may hold carriers no descendant probe covers.
/// - at least one live member exists (see [`live_closure_members`]).
/// - every live member resolves every property in `props`: an equality index
///   on `(member, prop)`, or `prop` being that member's id field (canonical
///   `id`, or its `id_field_aliases` entry). Coverage must be total —
///   a partial union would silently drop rows.
///
/// `props` are property *names*, because EXPLAIN sees an AST, not resolved
/// `PropertyMatcher` values, and because eligibility does not depend on the
/// value being probed.
pub(crate) fn closure_probe_members(
    graph: &DirGraph,
    node_type: &str,
    props: &[&str],
) -> Option<Vec<String>> {
    if props.is_empty() || !graph.managed_label_closed(node_type) {
        return None;
    }
    let members = live_closure_members(graph, node_type);
    if members.is_empty() {
        return None;
    }
    let covered = members.iter().all(|member| {
        props
            .iter()
            .all(|prop| member_covers_property(graph, member, prop))
    });
    covered.then(|| members.into_iter().map(str::to_string).collect())
}

/// A point lookup of `prop` on `member` resolves without a scan.
fn member_covers_property(graph: &DirGraph, member: &str, prop: &str) -> bool {
    if prop == "id" {
        return true;
    }
    if graph
        .id_field_aliases
        .get(member)
        .is_some_and(|alias| alias == prop)
    {
        return true;
    }
    graph.has_index(member, prop)
}
