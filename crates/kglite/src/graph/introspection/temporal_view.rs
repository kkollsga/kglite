//! The validity-interval attributes `describe()` prints on a node type's
//! `<type>` element and a relationship type's `<conn>` element.
//!
//! A config that is closed, unkeyed and was written without a declare-time
//! count prints exactly `temporal_from=".." temporal_to=".."`, the form every
//! earlier release printed. A declaration adds what distinguishes it:
//! `temporal_convention`, `temporal_source` and `temporal_abutting` (rows
//! whose end met another row's start when it was declared). A relationship
//! type with several declarations prints them once each in one `temporal`
//! attribute, since an element cannot repeat an attribute, in lookup order:
//! the source-keyed ones, then the unkeyed one as `other sources`, which is
//! what a relationship whose source has no keyed declaration uses.

use super::describe::xml_escape;
use crate::graph::dir_graph::DirGraph;
use crate::graph::features::temporal::TemporalTarget;
use crate::graph::schema::TemporalConfig;

fn attributes(config: &TemporalConfig, abutting: Option<usize>) -> String {
    let mut out = format!(
        " temporal_from=\"{}\" temporal_to=\"{}\"",
        xml_escape(&config.valid_from),
        xml_escape(&config.valid_to)
    );
    if !config.convention.is_closed() {
        out.push_str(&format!(
            " temporal_convention=\"{}\"",
            config.convention.as_str()
        ));
    }
    if let Some(source) = &config.source_type {
        out.push_str(&format!(" temporal_source=\"{}\"", xml_escape(source)));
    }
    if let Some(count) = abutting {
        out.push_str(&format!(" temporal_abutting=\"{count}\""));
    }
    out
}

/// `source: from..to[ half_open][ abutting=N]` — one declaration inside the
/// combined `temporal` attribute.
fn compact(config: &TemporalConfig, abutting: Option<usize>) -> String {
    let mut out = match &config.source_type {
        Some(source) => format!("{source}: "),
        None => "other sources: ".to_string(),
    };
    out.push_str(&format!("{}..{}", config.valid_from, config.valid_to));
    if !config.convention.is_closed() {
        out.push_str(&format!(" {}", config.convention.as_str()));
    }
    if let Some(count) = abutting {
        out.push_str(&format!(" abutting={count}"));
    }
    out
}

pub(super) fn node_attrs(graph: &DirGraph, label: &str) -> String {
    graph
        .temporal
        .node(label)
        .map_or_else(String::new, |config| {
            let target = TemporalTarget::Node(label.to_string());
            attributes(config, graph.temporal.abutting(&target))
        })
}

pub(super) fn conn_attrs(graph: &DirGraph, rel_type: &str) -> String {
    let counted = |config: &TemporalConfig| {
        graph.temporal.abutting(&TemporalTarget::Relationship {
            rel_type: rel_type.to_string(),
            source_type: config.source_type.clone(),
        })
    };
    match graph.temporal.edges(rel_type) {
        [] => String::new(),
        [only] => attributes(only, counted(only)),
        several => {
            let mut ordered: Vec<&TemporalConfig> = several.iter().collect();
            ordered.sort_by_key(|c| (c.source_type.is_none(), c.source_type.as_deref()));
            let listed = ordered
                .into_iter()
                .map(|config| compact(config, counted(config)))
                .collect::<Vec<_>>()
                .join("; ");
            format!(" temporal=\"{}\"", xml_escape(&listed))
        }
    }
}
