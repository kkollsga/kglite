//! Export-local portable paths shared by actual writes and blueprint references.

use std::collections::{BTreeMap, BTreeSet, HashMap};

const MAX_COMPONENT_BYTES: usize = 120;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Entry {
    Directory(String),
    Csv(String),
}

impl Entry {
    fn preferred(&self) -> String {
        match self {
            Self::Directory(name) => name.clone(),
            Self::Csv(name) => format!("{name}.csv"),
        }
    }

    fn fallback(&self, index: usize) -> String {
        match self {
            Self::Directory(_) => format!("export_{index}"),
            Self::Csv(_) => format!("export_{index}.csv"),
        }
    }
}

fn portable(component: &str) -> bool {
    if component.is_empty()
        || component.len() > MAX_COMPONENT_BYTES
        || matches!(component, "." | "..")
        || component.ends_with(['.', ' '])
        || !component
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b" _-.".contains(&b))
    {
        return false;
    }
    let stem = component
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}

/// Allocate one directory's complete entry set, reserving portable unambiguous
/// names before assigning fallback names. Files and directories share a namespace.
fn allocate(entries: impl IntoIterator<Item = Entry>) -> Result<BTreeMap<Entry, String>, String> {
    let entries: BTreeSet<_> = entries.into_iter().collect();
    let mut counts = BTreeMap::new();
    for entry in &entries {
        *counts
            .entry(entry.preferred().to_ascii_lowercase())
            .or_insert(0_usize) += 1;
    }
    let mut assigned = BTreeMap::new();
    let mut used = BTreeSet::new();
    for entry in &entries {
        let preferred = entry.preferred();
        let folded = preferred.to_ascii_lowercase();
        if portable(&preferred) && counts[&folded] == 1 {
            used.insert(folded);
            assigned.insert(entry.clone(), preferred);
        }
    }
    for entry in entries {
        if assigned.contains_key(&entry) {
            continue;
        }
        // n occupied names cannot occupy all n+1 distinct candidates.
        let selected = (0..=used.len())
            .map(|index| entry.fallback(index))
            .find(|name| !used.contains(&name.to_ascii_lowercase()))
            .ok_or_else(|| "Could not allocate a unique export filename".to_string())?;
        used.insert(selected.to_ascii_lowercase());
        assigned.insert(entry, selected);
    }
    Ok(assigned)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ExportPaths {
    nodes: BTreeMap<String, String>,
    connections: BTreeMap<String, String>,
    directories: BTreeSet<String>,
}

impl ExportPaths {
    pub(super) fn new<'a>(
        node_types: impl Iterator<Item = &'a String>,
        connection_types: impl Iterator<Item = &'a String>,
        parent_types: &HashMap<String, String>,
    ) -> Result<Self, String> {
        let nodes: BTreeSet<String> = node_types.cloned().collect();
        let parents: BTreeSet<String> = nodes
            .iter()
            .filter_map(|node| parent_types.get(node))
            .cloned()
            .collect();
        let root_entries = parents.iter().cloned().map(Entry::Directory).chain(
            nodes
                .iter()
                .filter(|node| !parent_types.contains_key(*node))
                .cloned()
                .map(Entry::Csv),
        );
        let root = allocate(root_entries)?;
        let mut paths = Self {
            nodes: BTreeMap::new(),
            connections: BTreeMap::new(),
            directories: BTreeSet::new(),
        };
        for node in nodes
            .iter()
            .filter(|node| !parent_types.contains_key(*node))
        {
            paths.nodes.insert(
                node.clone(),
                format!("nodes/{}", root[&Entry::Csv(node.clone())]),
            );
        }
        for parent in parents {
            let directory = format!("nodes/{}", root[&Entry::Directory(parent.clone())]);
            paths.directories.insert(directory.clone());
            let children = allocate(
                nodes
                    .iter()
                    .filter(|node| parent_types.get(*node) == Some(&parent))
                    .cloned()
                    .map(Entry::Csv),
            )?;
            for (entry, filename) in children {
                if let Entry::Csv(child) = entry {
                    paths.nodes.insert(child, format!("{directory}/{filename}"));
                }
            }
        }
        for (entry, filename) in allocate(connection_types.cloned().map(Entry::Csv))? {
            if let Entry::Csv(connection) = entry {
                paths
                    .connections
                    .insert(connection, format!("connections/{filename}"));
            }
        }
        Ok(paths)
    }

    pub(super) fn node(&self, node_type: &str) -> &str {
        &self.nodes[node_type]
    }

    pub(super) fn connection(&self, connection_type: &str) -> &str {
        &self.connections[connection_type]
    }

    pub(super) fn node_directories(&self) -> impl Iterator<Item = &str> {
        self.directories.iter().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Component, Path};

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn ordinary_names_and_parent_paths_are_preserved() {
        let nodes = strings(&["Person", "Child", "Company"]);
        let connections = strings(&["WORKS_AT"]);
        let parents = HashMap::from([("Child".into(), "Person".into())]);
        let paths = ExportPaths::new(nodes.iter(), connections.iter(), &parents).unwrap();
        assert_eq!(paths.node("Person"), "nodes/Person.csv");
        assert_eq!(paths.node("Child"), "nodes/Person/Child.csv");
        assert_eq!(paths.connection("WORKS_AT"), "connections/WORKS_AT.csv");
    }

    #[test]
    fn component_allocation_is_portable_unique_and_inventory_order_independent() {
        // Synthetic names only: no directories/files or external sentinels.
        let mut nodes = strings(&[
            "A",
            "a",
            "a/b",
            "a\\b",
            "..",
            "CON",
            "NUL.txt",
            "trailing ",
            "雪",
            "export_0",
        ]);
        nodes.push("x".repeat(300));
        let edges = nodes.clone();
        let first = ExportPaths::new(nodes.iter(), edges.iter(), &HashMap::new()).unwrap();
        nodes.reverse();
        let second = ExportPaths::new(nodes.iter(), edges.iter().rev(), &HashMap::new()).unwrap();
        assert_eq!(first, second);
        let paths: Vec<_> = first
            .nodes
            .values()
            .chain(first.connections.values())
            .collect();
        let unique: BTreeSet<_> = paths.iter().map(|path| path.to_ascii_lowercase()).collect();
        assert_eq!(paths.len(), unique.len());
        for path in paths {
            for component in Path::new(path).components() {
                let Component::Normal(component) = component else {
                    panic!("non-normal path component")
                };
                assert!(portable(component.to_str().unwrap()));
            }
        }
        assert_eq!(first.node("export_0"), "nodes/export_0.csv");
    }

    #[test]
    fn directory_and_csv_collisions_use_the_same_allocation() {
        let nodes = strings(&["Parent", "Parent.csv", "Child"]);
        let parents = HashMap::from([("Child".into(), "Parent.csv".into())]);
        let paths = ExportPaths::new(nodes.iter(), [].iter(), &parents).unwrap();
        let directory = paths.node_directories().next().unwrap();
        assert_ne!(
            paths.node("Parent").to_ascii_lowercase(),
            directory.to_ascii_lowercase()
        );
        assert_eq!(
            Path::new(paths.node("Child")).parent().unwrap(),
            Path::new(directory)
        );
    }

    #[test]
    fn selected_child_allocates_parent_directory_even_without_parent_rows() {
        let nodes = strings(&["Child"]);
        let parents = HashMap::from([("Child".into(), "Parent".into())]);
        let paths = ExportPaths::new(nodes.iter(), [].iter(), &parents).unwrap();
        assert_eq!(paths.node("Child"), "nodes/Parent/Child.csv");
        assert_eq!(
            paths.node_directories().collect::<Vec<_>>(),
            ["nodes/Parent"]
        );
    }
}
