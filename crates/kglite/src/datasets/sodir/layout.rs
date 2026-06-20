//! Filesystem layout for the Sodir workdir.
//!
//! ```text
//! path/
//!   sodir_index.json           per-dataset row counts + fetch timestamps
//!   blueprint_complement.json  (optional) persisted complement blueprint
//!   csv/{stem}.csv             cached dataset CSVs (flat directory)
//!   graph/                     built knowledge graph (disk mode only)
//! ```
//!
//! Sodir has only two storage modes — `memory` (rebuilt each open,
//! discarded) and `disk` (persisted under `graph/`).

use std::path::{Path, PathBuf};

/// Resolved paths for one Sodir workdir. Cheap to clone — wraps a
/// single `PathBuf`.
#[derive(Debug, Clone)]
pub struct Workdir {
    root: PathBuf,
}

impl Workdir {
    /// Wrap an existing or to-be-created workdir path. Does not touch
    /// the filesystem.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `csv/` — the flat cache of dataset CSVs.
    pub fn csv_dir(&self) -> PathBuf {
        self.root.join("csv")
    }

    /// `csv/{stem}.csv`.
    pub fn csv_path(&self, stem: &str) -> PathBuf {
        self.csv_dir().join(format!("{stem}.csv"))
    }

    /// `sodir_index.json` — the per-dataset fetch manifest.
    pub fn index_file(&self) -> PathBuf {
        self.root.join("sodir_index.json")
    }

    /// `blueprint_complement.json` — persisted complement blueprint.
    pub fn complement_file(&self) -> PathBuf {
        self.root.join("blueprint_complement.json")
    }

    /// `_compiled_blueprint.json` — scratch file holding the merged
    /// blueprint passed to `from_blueprint`.
    pub fn compiled_blueprint(&self) -> PathBuf {
        self.root.join("_compiled_blueprint.json")
    }

    /// `graph/` — the built graph (disk mode).
    pub fn graph_dir(&self) -> PathBuf {
        self.root.join("graph")
    }

    /// `graph/disk_graph_meta.json` — its mtime drives the disk-mode
    /// "reopen → load, don't rebuild" cooldown short-circuit.
    pub fn disk_graph_meta(&self) -> PathBuf {
        self.graph_dir().join("disk_graph_meta.json")
    }

    /// `graph/sodir_source.json` — build-time dataset snapshot.
    pub fn source_meta(&self) -> PathBuf {
        self.graph_dir().join("sodir_source.json")
    }

    /// True if a disk-mode graph already exists.
    pub fn graph_exists(&self) -> bool {
        self.disk_graph_meta().is_file()
    }

    /// Age in days of the disk-mode graph metadata, or `None` if no
    /// graph has been built. Drives the disk-mode cooldown short-circuit
    /// in `Workdir`-managed dataset fetchers. Lifted from kglite-py in
    /// 0.10.1.
    pub fn disk_graph_age_days(&self) -> Option<f64> {
        crate::datasets::sodir::index::file_mtime_age_days(&self.disk_graph_meta())
    }

    /// Create the `csv/` and `graph/` directories (idempotent).
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.csv_dir())?;
        std::fs::create_dir_all(self.graph_dir())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_well_formed() {
        let w = Workdir::new("/tmp/sodir");
        assert_eq!(w.csv_dir(), Path::new("/tmp/sodir/csv"));
        assert_eq!(w.csv_path("field"), Path::new("/tmp/sodir/csv/field.csv"));
        assert_eq!(w.index_file(), Path::new("/tmp/sodir/sodir_index.json"));
        assert_eq!(w.graph_dir(), Path::new("/tmp/sodir/graph"));
        assert_eq!(
            w.disk_graph_meta(),
            Path::new("/tmp/sodir/graph/disk_graph_meta.json")
        );
    }

    #[test]
    fn ensure_dirs_idempotent_and_graph_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let w = Workdir::new(tmp.path());
        assert!(!w.graph_exists());
        w.ensure_dirs().unwrap();
        w.ensure_dirs().unwrap(); // idempotent
        assert!(w.csv_dir().is_dir());
        assert!(w.graph_dir().is_dir());
        assert!(!w.graph_exists()); // dirs exist, but no disk_graph_meta.json
    }
}
