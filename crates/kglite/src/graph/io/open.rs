//! Shared graph open-or-create lifecycle used by server-style bindings.

use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::load_file;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

/// Open an existing graph, or create an empty graph in `create_mode` when the
/// path is absent.
///
/// Existing graphs always auto-detect their persisted storage mode. Passing
/// `None` deliberately makes a missing path an error, which lets command-line
/// bindings reject typos unless the operator explicitly opts into creation.
pub fn open_or_create_graph(
    path: &Path,
    create_mode: Option<StorageMode>,
) -> io::Result<Arc<DirGraph>> {
    if path.exists() {
        return load_file(&path.to_string_lossy());
    }

    let mode = create_mode.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "graph path '{}' does not exist and no creation storage mode was provided",
                path.display()
            ),
        )
    })?;
    let graph = new_dir_graph_in_mode(mode, Some(path))
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    Ok(Arc::new(graph))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::GraphRead;

    #[test]
    fn missing_path_requires_explicit_create_mode() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("missing.kgl");
        let err = open_or_create_graph(&missing, None)
            .err()
            .expect("missing path without create mode should fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("no creation storage mode"));
    }

    #[test]
    fn creates_requested_storage_mode_when_path_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let memory =
            open_or_create_graph(&tmp.path().join("memory.kgl"), Some(StorageMode::Memory))
                .unwrap();
        assert!(!memory.graph.is_mapped());
        assert!(!memory.graph.is_disk());

        let disk_path = tmp.path().join("disk");
        let disk = open_or_create_graph(&disk_path, Some(StorageMode::Disk)).unwrap();
        assert!(disk.graph.is_disk());
        assert!(disk_path.is_dir());
    }

    #[test]
    fn existing_graph_is_loaded_regardless_of_create_mode() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("existing.kgl");
        let mut graph = Arc::new(DirGraph::new());
        crate::graph::io::file::save_graph(&mut graph, &path.to_string_lossy()).unwrap();

        let loaded = open_or_create_graph(&path, Some(StorageMode::Disk)).unwrap();
        assert!(!loaded.graph.is_disk());
    }
}
