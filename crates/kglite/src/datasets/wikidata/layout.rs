//! Filesystem layout for the Wikidata workdir.
//!
//! The crate is responsible only for the *dump* tier — the cached
//! `.nt.bz2` file and its in-progress `.part`. The built graph
//! (`workdir/graph[_<N>m]/`) is the main `kglite` crate's concern, so
//! it is not modelled here.

use std::path::{Path, PathBuf};

/// The Wikimedia `latest-truthy` dump filename.
pub const DUMP_FILE: &str = "latest-truthy.nt.bz2";

/// Canonical URL of the upstream dump.
pub const DUMP_URL: &str = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2";

/// Resolved dump paths for one Wikidata workdir.
#[derive(Debug, Clone)]
pub struct Workdir {
    root: PathBuf,
}

impl Workdir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `latest-truthy.nt.bz2` — the cached dump.
    pub fn dump_path(&self) -> PathBuf {
        self.root.join(DUMP_FILE)
    }

    /// `latest-truthy.nt.bz2.part` — the in-progress resumable download.
    pub fn part_path(&self) -> PathBuf {
        self.root.join(format!("{DUMP_FILE}.part"))
    }

    /// Create the workdir directory (idempotent).
    pub fn ensure_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_well_formed() {
        let w = Workdir::new("/tmp/wd");
        assert_eq!(w.dump_path(), Path::new("/tmp/wd/latest-truthy.nt.bz2"));
        assert_eq!(
            w.part_path(),
            Path::new("/tmp/wd/latest-truthy.nt.bz2.part")
        );
    }
}
