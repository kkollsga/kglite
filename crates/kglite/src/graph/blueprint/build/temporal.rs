//! Validity intervals a blueprint declares through a spec's `temporal` key:
//! `{from, to, convention}` on a node spec, an `fk_edges` entry or a
//! `junction_edges` entry.
//!
//! [`check_temporal_specs`] reads the author's blueprint before anything loads:
//! it refuses a key that cannot be declared (an unknown convention, an edge
//! bound that is not one of the edge's stored properties) and warns where
//! nothing will be declared — a key without `convention`, or columns typed
//! `validFrom`/`validTo` with no key at all, which only type the column.
//!
//! [`declare_blueprint_temporal`] declares once every row is in: one pass at
//! the end of the build covers the buffered and streamed node and FK paths and
//! the chunked junctions alike, and validates the rows the graph finally
//! holds. A relationship is declared for the spec's node type as its source
//! type, so two specs writing one relationship type keep their own bounds.

use indexmap::IndexMap;

use super::specs::FlatSpec;
use super::BuildReport;
use crate::graph::blueprint::schema::{Blueprint, NodeSpec, TemporalSpec};
use crate::graph::dir_graph::DirGraph;
use crate::graph::features::temporal::{declare_loaded, IntervalConvention, TemporalTarget};

/// One `temporal` key (or the lack of one) on one spec, with the names the
/// spec stores its columns under.
struct SpecInterval<'a> {
    /// `node 'X'`, `fk_edge 'R' (node 'X')` or `junction 'R' (node 'X')`.
    place: String,
    temporal: Option<&'a TemporalSpec>,
    /// Edge only: the stored name of every column the edge carries, with the
    /// CSV name it comes from. `None` for a node, which keeps every column of
    /// its input under its own name.
    stored: Option<Vec<(String, String)>>,
    /// Columns typed `validFrom` / `validTo`, by stored name.
    typed_from: Option<String>,
    typed_to: Option<String>,
}

fn stored_edge_columns(
    properties: &[String],
    rename: &IndexMap<String, String>,
) -> Vec<(String, String)> {
    properties
        .iter()
        .map(|col| (rename.get(col).unwrap_or(col).clone(), col.clone()))
        .collect()
}

fn typed_role(
    types: &IndexMap<String, String>,
    role: &str,
    rename: &IndexMap<String, String>,
) -> Option<String> {
    types
        .iter()
        .find(|(_, ty)| ty.as_str() == role)
        .map(|(col, _)| rename.get(col).unwrap_or(col).clone())
}

fn spec_intervals<'a>(node_type: &str, spec: &'a NodeSpec) -> Vec<SpecInterval<'a>> {
    let none = IndexMap::new();
    let mut out = vec![SpecInterval {
        place: format!("node '{node_type}'"),
        temporal: spec.temporal.as_ref(),
        stored: None,
        typed_from: typed_role(&spec.properties, "validFrom", &none),
        typed_to: typed_role(&spec.properties, "validTo", &none),
    }];
    for (edge_type, fk) in &spec.connections.fk_edges {
        out.push(SpecInterval {
            place: format!("fk_edge '{edge_type}' (node '{node_type}')"),
            temporal: fk.temporal.as_ref(),
            stored: Some(stored_edge_columns(&fk.properties, &fk.rename)),
            typed_from: typed_role(&fk.property_types, "validFrom", &fk.rename),
            typed_to: typed_role(&fk.property_types, "validTo", &fk.rename),
        });
    }
    for (edge_type, junc) in &spec.connections.junction_edges {
        out.push(SpecInterval {
            place: format!("junction '{edge_type}' (node '{node_type}')"),
            temporal: junc.temporal.as_ref(),
            stored: Some(stored_edge_columns(&junc.properties, &junc.rename)),
            typed_from: typed_role(&junc.property_types, "validFrom", &junc.rename),
            typed_to: typed_role(&junc.property_types, "validTo", &junc.rename),
        });
    }
    out
}

/// The convention a `temporal` key names: `None` when it names none, an error
/// for a spelling that is neither `closed` nor `half_open`.
fn convention_of(
    place: &str,
    temporal: &TemporalSpec,
) -> Result<Option<IntervalConvention>, String> {
    let Some(text) = temporal.convention.as_deref() else {
        return Ok(None);
    };
    IntervalConvention::parse(text).map(Some).ok_or_else(|| {
        format!(
            "{place}: temporal convention '{text}' is not one of 'closed' (the `to` day is the \
             last valid day) or 'half_open' (the `to` day is the first day no longer valid)"
        )
    })
}

/// An edge's `from`/`to` must name one of the properties the edge stores —
/// after `rename`, since that is the name the bound is read under.
fn check_edge_bound(
    place: &str,
    bound: &str,
    side: &str,
    stored: &[(String, String)],
) -> Result<(), String> {
    if stored.iter().any(|(name, _)| name == bound) {
        return Ok(());
    }
    let hint = match stored.iter().find(|(_, csv)| csv == bound) {
        Some((name, _)) => format!(
            " '{bound}' is renamed to '{name}', and the bound names the stored property: \
             write '{side}': '{name}'."
        ),
        None => {
            let names: Vec<String> = stored.iter().map(|(n, _)| format!("'{n}'")).collect();
            format!(
                " The edge stores: {}.",
                if names.is_empty() {
                    "no properties".to_string()
                } else {
                    names.join(", ")
                }
            )
        }
    };
    Err(format!(
        "{place}: temporal '{side}' names '{bound}', which is not a property this edge stores \
         (list the column in 'properties').{hint}"
    ))
}

fn missing_convention_warning(place: &str, temporal: &TemporalSpec) -> String {
    format!(
        "{place}: temporal names no convention, so no validity interval is declared. Add \
         \"convention\": \"closed\" if the `to` day is the last valid day, or \"half_open\" if \
         it is the first day no longer valid (from '{}', to '{}').",
        temporal.from, temporal.to
    )
}

fn typed_only_warning(place: &str, from: Option<&str>, to: Option<&str>) -> String {
    format!(
        "{place}: columns typed validFrom/validTo only type the column as a date; no validity \
         interval is declared. To declare one, add \"temporal\": {{\"from\": \"{}\", \"to\": \
         \"{}\", \"convention\": \"closed\" or \"half_open\"}}.",
        from.unwrap_or("<from column>"),
        to.unwrap_or("<to column>")
    )
}

/// Check every spec's `temporal` key before the build loads anything, and
/// return the warnings for specs that will declare nothing.
pub(crate) fn check_temporal_specs(blueprint: &Blueprint) -> Result<Vec<String>, String> {
    fn walk(warnings: &mut Vec<String>, node_type: &str, spec: &NodeSpec) -> Result<(), String> {
        for interval in spec_intervals(node_type, spec) {
            let place = &interval.place;
            let Some(temporal) = interval.temporal else {
                if interval.typed_from.is_some() || interval.typed_to.is_some() {
                    warnings.push(typed_only_warning(
                        place,
                        interval.typed_from.as_deref(),
                        interval.typed_to.as_deref(),
                    ));
                }
                continue;
            };
            if temporal.from == temporal.to {
                return Err(format!(
                    "{place}: temporal 'from' and 'to' both name '{}'; they must be two \
                     properties",
                    temporal.from
                ));
            }
            if let Some(stored) = &interval.stored {
                check_edge_bound(place, &temporal.from, "from", stored)?;
                check_edge_bound(place, &temporal.to, "to", stored)?;
            }
            if convention_of(place, temporal)?.is_none() {
                warnings.push(missing_convention_warning(place, temporal));
            }
        }
        for (sub_type, sub) in &spec.sub_nodes {
            walk(warnings, sub_type, sub)?;
        }
        Ok(())
    }
    let mut warnings = Vec::new();
    for (node_type, spec) in &blueprint.nodes {
        walk(&mut warnings, node_type, spec)?;
    }
    Ok(warnings)
}

/// Declare every interval the specs name a convention for, over the rows the
/// build wrote. A declaration the rows refuse — an unreadable bound, or an
/// inverted or empty interval — fails the build, naming the row. A spec that
/// wrote no rows of its target declares nothing and says so.
pub(super) fn declare_blueprint_temporal(
    graph: &mut DirGraph,
    all_specs: &[&FlatSpec],
    report: &mut BuildReport,
) -> Result<(), String> {
    for flat in all_specs {
        let node_type = flat.node_type.as_str();
        let spec = &flat.spec;
        let edges = spec
            .connections
            .fk_edges
            .iter()
            .map(|(rel, fk)| (rel, fk.temporal.as_ref(), "fk_edge"))
            .chain(
                spec.connections
                    .junction_edges
                    .iter()
                    .map(|(rel, j)| (rel, j.temporal.as_ref(), "junction")),
            );
        let mut requests: Vec<(String, TemporalTarget, &TemporalSpec)> = Vec::new();
        if let Some(temporal) = &spec.temporal {
            requests.push((
                format!("node '{node_type}'"),
                TemporalTarget::Node(node_type.to_string()),
                temporal,
            ));
        }
        for (rel_type, temporal, kind) in edges {
            if let Some(temporal) = temporal {
                requests.push((
                    format!("{kind} '{rel_type}' (node '{node_type}')"),
                    TemporalTarget::Relationship {
                        rel_type: rel_type.clone(),
                        source_type: Some(node_type.to_string()),
                    },
                    temporal,
                ));
            }
        }
        for (place, target, temporal) in requests {
            // `check_temporal_specs` refused an unknown spelling and warned
            // about a missing one before the build began.
            let Some(convention) = convention_of(&place, temporal)? else {
                continue;
            };
            declare_one(graph, &place, &target, temporal, convention, report)?;
        }
    }
    Ok(())
}

fn declare_one(
    graph: &mut DirGraph,
    place: &str,
    target: &TemporalTarget,
    temporal: &TemporalSpec,
    convention: IntervalConvention,
    report: &mut BuildReport,
) -> Result<(), String> {
    let (present, written): (bool, Vec<&str>) = match target {
        // A node load registers every column it writes, so an unknown bound
        // is left for the declaration to refuse.
        TemporalTarget::Node(label) => (graph.has_node_type(label), Vec::new()),
        // An edge registers a property only once some row holds a value, so
        // a bound every row left open would read as missing. The bounds were
        // checked against the edge's listed properties before the build.
        TemporalTarget::Relationship {
            rel_type,
            source_type,
        } => (
            graph
                .connection_type_metadata
                .get(rel_type)
                .zip(source_type.as_ref())
                .is_some_and(|(info, source)| info.source_types.contains(source)),
            vec![temporal.from.as_str(), temporal.to.as_str()],
        ),
    };
    if !present {
        report.warnings.push(format!(
            "{place}: the build wrote no rows of {}, so no validity interval is declared",
            target.describe()
        ));
        return Ok(());
    }
    let declared = declare_loaded(
        graph,
        target,
        &temporal.from,
        &temporal.to,
        convention,
        &written,
    )
    .map_err(|reason| format!("{place}: the temporal declaration is refused: {reason}"))?;
    report.warnings.extend(declared.warning);
    Ok(())
}

#[cfg(test)]
#[path = "temporal_tests.rs"]
mod tests;
