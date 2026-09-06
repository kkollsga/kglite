//! Per-invocation output ownership: writes and blueprint references share one map.
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use super::super::schema::{Blueprint, ComputeOp, NodeSpec};
use super::{resolve_input_path, sanitize_filename};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Output {
    Derive(String),
    Filter(String),
    Aggregate(String),
    Chain(String, String),
    CalendarNode(String),
    CalendarNext(String, String),
    CalendarHierarchy(String, String, String),
    CalendarLink(String, String, String),
}

impl Output {
    fn preferred(&self) -> String {
        let clean = sanitize_filename;
        match self {
            Self::Derive(name) => format!("{}_derived.csv", clean(name)),
            Self::Filter(name) => format!("{}_filtered.csv", clean(name)),
            Self::Aggregate(name) => format!("aggregate_{}.csv", clean(name)),
            Self::Chain(_, edge) => format!("chain_{}.csv", clean(edge)),
            Self::CalendarNode(name) => format!("calendar_{}.csv", clean(name)),
            Self::CalendarNext(name, edge) | Self::CalendarHierarchy(name, _, edge) => {
                format!("calendar_{}_{}.csv", clean(name), clean(edge))
            }
            Self::CalendarLink(from, _, edge) => {
                format!("calendar_link_{}_{}.csv", clean(from), clean(edge))
            }
        }
    }
}

fn outputs(ops: &[ComputeOp]) -> BTreeSet<Output> {
    let mut entries = BTreeSet::new();
    for op in ops {
        match op {
            ComputeOp::Derive { from, .. } => {
                entries.insert(Output::Derive(from.clone()));
            }
            ComputeOp::Filter { from, into, .. } => {
                entries.insert(Output::Filter(into.as_ref().unwrap_or(from).clone()));
            }
            ComputeOp::Aggregate { into, .. } => {
                entries.insert(Output::Aggregate(into.clone()));
            }
            ComputeOp::Chain { from, edge, .. } => {
                entries.insert(Output::Chain(from.clone(), edge.clone()));
            }
            ComputeOp::Calendar {
                node_type,
                next_edge,
                in_month_edge,
                in_quarter_edge,
                links,
                ..
            } => {
                entries.insert(Output::CalendarNode(node_type.clone()));
                entries.insert(Output::CalendarNext(node_type.clone(), next_edge.clone()));
                for (hierarchy, edge) in [("Month", in_month_edge), ("Quarter", in_quarter_edge)] {
                    if let Some(edge) = edge {
                        entries.insert(Output::CalendarNode(hierarchy.to_string()));
                        entries.insert(Output::CalendarHierarchy(
                            node_type.clone(),
                            hierarchy.to_string(),
                            edge.clone(),
                        ));
                    }
                }
                for link in links {
                    entries.insert(Output::CalendarLink(
                        link.from.clone(),
                        node_type.clone(),
                        link.edge.clone(),
                    ));
                }
            }
        }
    }
    entries
}

fn normalized(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

fn reserve_source(used: &mut BTreeSet<String>, root: &Path, computed: &Path, source: &str) {
    let path = normalized(&resolve_input_path(root, source));
    if path
        .parent()
        .is_some_and(|parent| normalized(parent) == computed)
    {
        if let Some(name) = path.file_name() {
            used.insert(name.to_string_lossy().to_lowercase());
        }
    }
}

fn reserve_spec(used: &mut BTreeSet<String>, root: &Path, computed: &Path, spec: &NodeSpec) {
    if let Some(csv) = &spec.csv {
        reserve_source(used, root, computed, csv);
    }
    for edge in spec.connections.junction_edges.values() {
        if let Some(csv) = &edge.csv {
            reserve_source(used, root, computed, csv);
        }
    }
    for child in spec.sub_nodes.values() {
        reserve_spec(used, root, computed, child);
    }
}

struct HierarchyState {
    keys: BTreeSet<String>,
    pk: String,
}

pub(super) struct ComputePaths(
    BTreeMap<Output, String>,
    RefCell<BTreeMap<String, HierarchyState>>,
);

impl ComputePaths {
    /// Reserve all active inputs and existing directory entries, including unknown
    /// prior outputs. Only this invocation's same logical output may replace itself.
    pub(super) fn new(
        blueprint: &Blueprint,
        root: &Path,
        ops: &[ComputeOp],
    ) -> Result<Self, String> {
        let root = normalized(root);
        let computed = normalized(&root.join("computed"));
        let mut used = BTreeSet::new();
        if computed.exists() {
            for entry in std::fs::read_dir(&computed)
                .map_err(|e| format!("compute: inspect {}: {e}", computed.display()))?
            {
                let entry = entry.map_err(|e| format!("compute: inspect output: {e}"))?;
                used.insert(entry.file_name().to_string_lossy().to_lowercase());
            }
        }
        for file in blueprint.files.values() {
            if let Some(path) = &file.path {
                reserve_source(&mut used, &root, &computed, path);
            }
        }
        if let Some(destination) = blueprint.settings.resolved_output(&root) {
            reserve_source(&mut used, &root, &computed, &destination.to_string_lossy());
        }
        for spec in blueprint.nodes.values() {
            reserve_spec(&mut used, &root, &computed, spec);
        }
        let entries = outputs(ops);
        let mut counts = BTreeMap::new();
        for entry in &entries {
            *counts
                .entry(entry.preferred().to_ascii_lowercase())
                .or_insert(0_usize) += 1;
        }
        let mut assigned = BTreeMap::new();
        for entry in &entries {
            let preferred = entry.preferred();
            let folded = preferred.to_ascii_lowercase();
            if preferred.len() <= 120
                && counts[&folded] == 1
                && !used.contains(&folded)
                && computed.join(&preferred).symlink_metadata().is_err()
            {
                used.insert(folded);
                assigned.insert(entry.clone(), format!("computed/{preferred}"));
            }
        }
        for entry in entries {
            if assigned.contains_key(&entry) {
                continue;
            }
            // n occupied names cannot occupy all n+1 distinct fallback names.
            let name = (0..=used.len())
                .map(|i| format!("compute_{i}.csv"))
                .find(|name| {
                    !used.contains(&name.to_ascii_lowercase())
                        && computed.join(name).symlink_metadata().is_err()
                })
                .ok_or_else(|| {
                    "compute: could not allocate a unique output filename".to_string()
                })?;
            used.insert(name.to_ascii_lowercase());
            assigned.insert(entry, format!("computed/{name}"));
        }
        Ok(Self(assigned, RefCell::new(BTreeMap::new())))
    }

    pub(super) fn owns_hierarchy(&self, name: &str, spec: &NodeSpec) -> bool {
        let owned = self.1.borrow();
        let Some(state) = owned.get(name) else {
            return false;
        };
        // Exhaustive fields pin the generated contract: derive/filter/chain changes
        // invalidate ownership rather than being silently discarded by a later calendar.
        let NodeSpec {
            csv,
            file,
            pk,
            title,
            parent,
            parent_fk,
            properties,
            labels,
            skipped,
            filter,
            connections,
            sub_nodes,
            timeseries,
            extra,
        } = spec;
        csv.as_deref()
            == Some(
                self.relative(&Output::CalendarNode(name.to_string()))
                    .as_str(),
            )
            && pk.as_deref() == Some(state.pk.as_str())
            && title == pk
            && file.is_none()
            && parent.is_none()
            && parent_fk.is_none()
            && properties.is_empty()
            && labels.is_empty()
            && skipped.is_empty()
            && filter.is_empty()
            && connections.fk_edges.is_empty()
            && connections.junction_edges.is_empty()
            && sub_nodes.is_empty()
            && timeseries.is_none()
            && extra.is_empty()
    }

    pub(super) fn hierarchy_keys(&self, name: &str, keys: Vec<String>) -> Vec<String> {
        let mut union = self
            .1
            .borrow()
            .get(name)
            .map(|state| state.keys.clone())
            .unwrap_or_default();
        union.extend(keys);
        union.into_iter().collect()
    }

    pub(super) fn publish_hierarchy_keys(&self, name: &str, pk: &str, keys: Vec<String>) {
        self.1.borrow_mut().insert(
            name.to_string(),
            HierarchyState {
                keys: keys.into_iter().collect(),
                pk: pk.to_string(),
            },
        );
    }

    pub(super) fn relative(&self, output: &Output) -> String {
        self.0
            .get(output)
            .expect("compute output missing from complete allocation")
            .clone()
    }
}

#[cfg(test)]
#[path = "paths_tests.rs"]
mod tests;
