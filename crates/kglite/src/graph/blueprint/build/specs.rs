//! Spec flattening: one `FlatSpec` per node type, with each spec's
//! `sub_nodes` lifted out into `FlatSpec`s of their own.

use super::super::schema::NodeSpec;
use indexmap::IndexMap;

/// Flattened view of one node spec with parent info carried along.
pub struct FlatSpec {
    pub node_type: String,
    pub spec: NodeSpec,
    pub parent: Option<String>,
    pub is_manual: bool,
}

pub(super) fn collect_specs(nodes: &IndexMap<String, NodeSpec>) -> (Vec<FlatSpec>, Vec<FlatSpec>) {
    let mut core = Vec::new();
    let mut subs = Vec::new();
    for (name, spec) in nodes {
        let is_manual = spec.csv.is_none();
        core.push(FlatSpec {
            node_type: name.clone(),
            spec: clone_without_subs(spec),
            parent: None,
            is_manual,
        });
        for (sub_name, sub_spec) in &spec.sub_nodes {
            // Sub-nodes keep their raw `parent` field untouched — the
            // enclosing type name is recorded on `FlatSpec.parent` so we
            // can call `set_parent_type` without also generating an
            // implicit OF_PARENT edge (that is reserved for top-level
            // specs that explicitly declare `parent` + `parent_fk`).
            let sub_clone = clone_without_subs(sub_spec);
            subs.push(FlatSpec {
                node_type: sub_name.clone(),
                spec: sub_clone,
                parent: Some(name.clone()),
                is_manual: false,
            });
        }
    }
    (core, subs)
}

/// The flattening pass's per-type copy: everything the spec declares except
/// its `sub_nodes`, which are flattened into their own `FlatSpec`s.
///
/// Struct-update syntax on purpose — a field-by-field copy silently drops any
/// field added to `NodeSpec` later, and the loss shows up as a directive the
/// blueprint declares and the build ignores.
fn clone_without_subs(spec: &NodeSpec) -> NodeSpec {
    NodeSpec {
        sub_nodes: IndexMap::new(),
        ..spec.clone()
    }
}

#[cfg(test)]
#[path = "../build_spec_clone_tests.rs"]
mod spec_clone_tests;
