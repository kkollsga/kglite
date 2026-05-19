//! Filesystem layout for the SEC workdir.
//!
//! The workdir is a strict three-tier cache:
//!
//! ```text
//! path/
//!   raw/              tier 1: immutable byte-for-byte SEC cache
//!   processed/        tier 2: parsed CSVs (shared across modes)
//!   graph/{mode}/     tier 3: built knowledge graph, one subdir per mode
//! ```
//!
//! Opening with `mode=X` never touches `graph/Y/`. See the plan doc
//! for the full idempotency contract.

use std::fmt;
use std::path::{Path, PathBuf};

/// Storage mode for the built graph. Each mode lives in its own
/// `graph/{mode}/` subdirectory; modes coexist freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageMode {
    Memory,
    Mapped,
    Disk,
}

impl StorageMode {
    pub fn as_str(self) -> &'static str {
        match self {
            StorageMode::Memory => "memory",
            StorageMode::Mapped => "mapped",
            StorageMode::Disk => "disk",
        }
    }
}

impl fmt::Display for StorageMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for StorageMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "memory" => Ok(StorageMode::Memory),
            "mapped" => Ok(StorageMode::Mapped),
            "disk" => Ok(StorageMode::Disk),
            other => Err(format!(
                "unknown storage mode '{other}'; expected memory|mapped|disk"
            )),
        }
    }
}

/// Resolved paths for one workdir. Cheap to clone — wraps a single
/// `PathBuf`.
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

    // ── raw/ tier ─────────────────────────────────────────────────────

    pub fn raw_dir(&self) -> PathBuf {
        self.root.join("raw")
    }

    pub fn raw_index_dir(&self) -> PathBuf {
        self.raw_dir().join("index")
    }

    /// `raw/index/master.{year}_QTR{n}.idx`
    pub fn raw_master_idx(&self, year: u16, quarter: u8) -> PathBuf {
        self.raw_index_dir()
            .join(format!("master.{year}_QTR{quarter}.idx"))
    }

    pub fn raw_submissions_dir(&self) -> PathBuf {
        self.raw_dir().join("submissions")
    }

    /// `raw/submissions/submissions.zip` — the nightly bulk.
    pub fn raw_submissions_zip(&self) -> PathBuf {
        self.raw_submissions_dir().join("submissions.zip")
    }

    pub fn raw_insider_dir(&self) -> PathBuf {
        self.raw_dir().join("insider")
    }

    pub fn raw_form13f_dir(&self) -> PathBuf {
        self.raw_dir().join("form13f")
    }

    pub fn raw_financials_dir(&self) -> PathBuf {
        self.raw_dir().join("financials")
    }

    pub fn raw_filings_dir(&self) -> PathBuf {
        self.raw_dir().join("filings")
    }

    pub fn raw_company_tickers_json(&self) -> PathBuf {
        self.raw_dir().join("company_tickers.json")
    }

    /// Download log: `raw/raw_manifest.json`.
    pub fn raw_manifest(&self) -> PathBuf {
        self.raw_dir().join("raw_manifest.json")
    }

    // ── processed/ tier ───────────────────────────────────────────────

    pub fn processed_dir(&self) -> PathBuf {
        self.root.join("processed")
    }

    pub fn processed_csv(&self, name: &str) -> PathBuf {
        self.processed_dir().join(format!("{name}.csv"))
    }

    pub fn processed_manifest(&self) -> PathBuf {
        self.processed_dir().join("processed_manifest.json")
    }

    // ── graph/{mode}/ tier ────────────────────────────────────────────

    pub fn graph_dir(&self, mode: StorageMode) -> PathBuf {
        self.root.join("graph").join(mode.as_str())
    }

    pub fn graph_kgl(&self, mode: StorageMode) -> PathBuf {
        self.graph_dir(mode).join("sec.kgl")
    }

    pub fn graph_manifest(&self, mode: StorageMode) -> PathBuf {
        self.graph_dir(mode).join("graph_manifest.json")
    }

    /// True if a graph for the given mode already exists. Caller uses
    /// this for the "reopen → load, don't rebuild" contract.
    pub fn graph_exists(&self, mode: StorageMode) -> bool {
        match mode {
            // Memory/Mapped: a single .kgl file.
            StorageMode::Memory | StorageMode::Mapped => self.graph_kgl(mode).is_file(),
            // Disk: a directory of mmap files with a manifest.
            StorageMode::Disk => self.graph_manifest(mode).is_file(),
        }
    }

    /// Create the tier directories (idempotent).
    pub fn ensure_dirs(&self, mode: Option<StorageMode>) -> std::io::Result<()> {
        std::fs::create_dir_all(self.raw_index_dir())?;
        std::fs::create_dir_all(self.raw_submissions_dir())?;
        std::fs::create_dir_all(self.raw_insider_dir())?;
        std::fs::create_dir_all(self.raw_form13f_dir())?;
        std::fs::create_dir_all(self.raw_financials_dir())?;
        std::fs::create_dir_all(self.raw_filings_dir())?;
        std::fs::create_dir_all(self.processed_dir())?;
        if let Some(m) = mode {
            std::fs::create_dir_all(self.graph_dir(m))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_well_formed() {
        let w = Workdir::new("/tmp/sec");
        assert_eq!(w.raw_dir(), Path::new("/tmp/sec/raw"));
        assert_eq!(
            w.raw_master_idx(2024, 4),
            Path::new("/tmp/sec/raw/index/master.2024_QTR4.idx")
        );
        assert_eq!(
            w.raw_submissions_zip(),
            Path::new("/tmp/sec/raw/submissions/submissions.zip")
        );
        assert_eq!(
            w.processed_csv("company"),
            Path::new("/tmp/sec/processed/company.csv")
        );
        assert_eq!(
            w.graph_kgl(StorageMode::Mapped),
            Path::new("/tmp/sec/graph/mapped/sec.kgl")
        );
        assert_eq!(
            w.graph_dir(StorageMode::Disk),
            Path::new("/tmp/sec/graph/disk")
        );
    }

    #[test]
    fn storage_mode_roundtrip() {
        for m in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
            let parsed: StorageMode = m.as_str().parse().unwrap();
            assert_eq!(parsed, m);
        }
        assert!("bogus".parse::<StorageMode>().is_err());
    }

    #[test]
    fn ensure_dirs_idempotent() {
        let tmp = tempdir();
        let w = Workdir::new(&tmp);
        w.ensure_dirs(Some(StorageMode::Mapped)).unwrap();
        w.ensure_dirs(Some(StorageMode::Mapped)).unwrap(); // idempotent
        assert!(w.raw_index_dir().is_dir());
        assert!(w.processed_dir().is_dir());
        assert!(w.graph_dir(StorageMode::Mapped).is_dir());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn graph_exists_false_for_empty_workdir() {
        let tmp = tempdir();
        let w = Workdir::new(&tmp);
        assert!(!w.graph_exists(StorageMode::Memory));
        assert!(!w.graph_exists(StorageMode::Mapped));
        assert!(!w.graph_exists(StorageMode::Disk));
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Create an isolated tempdir under the OS temp directory.
    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kglite-sec-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
