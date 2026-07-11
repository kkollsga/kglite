//! Immutable disk-generation publication and writer ownership.

use fs2::FileExt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const CURRENT_FILE: &str = "CURRENT";
const GENERATIONS_DIR: &str = "generations";
const GENERATION_PREFIX: &str = "gen_";
const STAGE_PREFIX: &str = ".stage-";
static NEXT_WORKSPACE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(test)]
thread_local! {
    static PUBLISH_FAILPOINT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

fn publish_failpoint(stage: &'static str) -> io::Result<()> {
    #[cfg(test)]
    if PUBLISH_FAILPOINT.with(|point| point.get() == Some(stage)) {
        return Err(io::Error::other(format!(
            "injected generation publish failure at {stage}"
        )));
    }
    let _ = stage;
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedSnapshot {
    pub(crate) logical_root: PathBuf,
    pub(crate) snapshot_dir: PathBuf,
    pub(crate) generation: Option<u64>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn generation_name(id: u64) -> String {
    format!("{GENERATION_PREFIX}{id:020}")
}

fn parse_generation_name(name: &str) -> io::Result<u64> {
    let digits = name
        .strip_prefix(GENERATION_PREFIX)
        .ok_or_else(|| invalid("CURRENT does not name a KGLite generation"))?;
    if digits.len() != 20 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid("CURRENT contains an invalid generation name"));
    }
    digits
        .parse()
        .map_err(|_| invalid("CURRENT generation number is out of range"))
}

/// Resolve the immutable snapshot selected by `CURRENT`. A missing pointer is
/// the legacy flat-directory format; a present but invalid pointer is always
/// an error and never falls back to possibly stale legacy files.
pub(crate) fn resolve_snapshot(root: &Path) -> io::Result<ResolvedSnapshot> {
    let current = root.join(CURRENT_FILE);
    if !current.exists() {
        return Ok(ResolvedSnapshot {
            logical_root: root.to_path_buf(),
            snapshot_dir: root.to_path_buf(),
            generation: None,
        });
    }
    let raw = fs::read_to_string(&current)?;
    let name = raw
        .strip_suffix('\n')
        .ok_or_else(|| invalid("CURRENT must end with one newline"))?;
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(invalid("CURRENT contains a path component"));
    }
    let generation = parse_generation_name(name)?;
    let snapshot_dir = root.join(GENERATIONS_DIR).join(name);
    if !snapshot_dir.is_dir() || !snapshot_dir.join("metadata.json").is_file() {
        return Err(invalid(format!(
            "CURRENT selects incomplete or missing generation {name}"
        )));
    }
    Ok(ResolvedSnapshot {
        logical_root: root.to_path_buf(),
        snapshot_dir,
        generation: Some(generation),
    })
}

/// Owned advisory writer lease. Readers do not take this lock: they resolve
/// `CURRENT` once and keep their immutable mmap generation alive.
#[derive(Debug)]
pub(crate) struct GraphDirectoryLock {
    _file: File,
    pub(crate) root: PathBuf,
}

#[derive(Debug)]
pub(crate) struct MutationWorkspace {
    root: PathBuf,
    segment: PathBuf,
}

impl MutationWorkspace {
    pub(crate) fn create(graph_root: &Path) -> io::Result<Self> {
        let nonce = NEXT_WORKSPACE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = graph_root.join(format!(".working-{}-{nonce:x}", std::process::id()));
        let segment = root.join("seg_000");
        fs::create_dir_all(&segment)?;
        Ok(Self { root, segment })
    }

    pub(crate) fn segment_dir(&self) -> &Path {
        &self.segment
    }
}

impl Drop for MutationWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

impl GraphDirectoryLock {
    pub(crate) fn try_acquire(root: &Path) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        let lock_path = root.join(".kglite.lock");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        file.try_lock_exclusive().map_err(|error| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "disk graph {} already has an active writer: {error}",
                    root.display()
                ),
            )
        })?;
        file.set_len(0)?;
        writeln!(file, "pid={}", std::process::id())?;
        file.sync_all()?;
        Ok(Self {
            _file: file,
            root: root.to_path_buf(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct GenerationTxn {
    root: PathBuf,
    generations: PathBuf,
    stage: PathBuf,
    final_dir: PathBuf,
    name: String,
}

impl GenerationTxn {
    pub(crate) fn begin(root: &Path) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        let generations = root.join(GENERATIONS_DIR);
        fs::create_dir_all(&generations)?;
        for entry in fs::read_dir(&generations)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(STAGE_PREFIX) && entry.file_type()?.is_dir() {
                fs::remove_dir_all(entry.path())?;
            }
        }
        let mut max_id = resolve_snapshot(root)?.generation.unwrap_or(0);
        for entry in fs::read_dir(&generations)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                if let Ok(id) = parse_generation_name(name) {
                    max_id = max_id.max(id);
                }
            }
        }
        let id = max_id
            .checked_add(1)
            .ok_or_else(|| invalid("generation counter exhausted"))?;
        let name = generation_name(id);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let stage = generations.join(format!("{STAGE_PREFIX}{}-{nonce:x}", std::process::id()));
        fs::create_dir(&stage)?;
        let final_dir = generations.join(&name);
        Ok(Self {
            root: root.to_path_buf(),
            generations,
            stage,
            final_dir,
            name,
        })
    }

    pub(crate) fn stage_dir(&self) -> &Path {
        &self.stage
    }

    pub(crate) fn publish(self) -> io::Result<PathBuf> {
        if !self.stage.join("metadata.json").is_file()
            || !self.stage.join("disk_graph_meta.json").is_file()
        {
            return Err(invalid("generation stage is missing completion metadata"));
        }
        sync_tree(&self.stage)?;
        publish_failpoint("before_generation_rename")?;
        fs::rename(&self.stage, &self.final_dir)?;
        sync_directory(&self.generations)?;
        publish_failpoint("after_generation_rename")?;

        let mut pointer = tempfile::NamedTempFile::new_in(&self.root)?;
        writeln!(pointer, "{}", self.name)?;
        pointer.flush()?;
        pointer.as_file().sync_all()?;
        publish_failpoint("before_current_replace")?;
        pointer
            .persist(self.root.join(CURRENT_FILE))
            .map_err(|e| e.error)?;
        publish_failpoint("after_current_replace")?;
        sync_directory(&self.root)?;
        Ok(self.final_dir)
    }
}

fn sync_tree(root: &Path) -> io::Result<()> {
    let mut dirs = Vec::new();
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = entry.map_err(io::Error::other)?;
        if entry.file_type().is_file() {
            File::open(entry.path())?.sync_all()?;
        } else if entry.file_type().is_dir() {
            dirs.push(entry.path().to_path_buf());
        }
    }
    for dir in dirs.into_iter().rev() {
        sync_directory(&dir)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inject(stage: &'static str, action: impl FnOnce()) {
        PUBLISH_FAILPOINT.with(|point| point.set(Some(stage)));
        action();
        PUBLISH_FAILPOINT.with(|point| point.set(None));
    }

    fn complete_stage(txn: &GenerationTxn) {
        fs::write(txn.stage_dir().join("metadata.json"), b"{}").unwrap();
        fs::write(txn.stage_dir().join("disk_graph_meta.json"), b"{}").unwrap();
    }

    #[test]
    fn legacy_and_current_resolution_are_strict() {
        let root = tempfile::tempdir().unwrap();
        let legacy = resolve_snapshot(root.path()).unwrap();
        assert_eq!(legacy.snapshot_dir, root.path());
        assert_eq!(legacy.generation, None);

        fs::write(root.path().join(CURRENT_FILE), "../escape\n").unwrap();
        assert_eq!(
            resolve_snapshot(root.path()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn publish_selects_only_complete_generation_and_cleans_stages() {
        let root = tempfile::tempdir().unwrap();
        let lock = GraphDirectoryLock::try_acquire(root.path()).unwrap();
        let abandoned = root.path().join(GENERATIONS_DIR).join(".stage-old");
        fs::create_dir_all(&abandoned).unwrap();
        fs::write(abandoned.join("junk"), b"x").unwrap();

        let txn = GenerationTxn::begin(root.path()).unwrap();
        assert!(!abandoned.exists());
        complete_stage(&txn);
        let published = txn.publish().unwrap();
        let resolved = resolve_snapshot(root.path()).unwrap();
        assert_eq!(resolved.snapshot_dir, published);
        assert_eq!(resolved.generation, Some(1));
        drop(lock);
    }

    #[test]
    fn second_writer_is_rejected_until_first_drops() {
        let root = tempfile::tempdir().unwrap();
        let first = GraphDirectoryLock::try_acquire(root.path()).unwrap();
        assert_eq!(
            GraphDirectoryLock::try_acquire(root.path())
                .unwrap_err()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        drop(first);
        GraphDirectoryLock::try_acquire(root.path()).unwrap();
    }

    #[test]
    fn failures_before_pointer_keep_old_and_after_pointer_select_new() {
        let root = tempfile::tempdir().unwrap();
        let _lock = GraphDirectoryLock::try_acquire(root.path()).unwrap();
        let first = GenerationTxn::begin(root.path()).unwrap();
        complete_stage(&first);
        first.publish().unwrap();

        for stage in [
            "before_generation_rename",
            "after_generation_rename",
            "before_current_replace",
        ] {
            let before = resolve_snapshot(root.path()).unwrap().generation;
            let txn = GenerationTxn::begin(root.path()).unwrap();
            complete_stage(&txn);
            inject(stage, || assert!(txn.publish().is_err()));
            assert_eq!(resolve_snapshot(root.path()).unwrap().generation, before);
        }

        let before = resolve_snapshot(root.path()).unwrap().generation.unwrap();
        let txn = GenerationTxn::begin(root.path()).unwrap();
        complete_stage(&txn);
        inject("after_current_replace", || assert!(txn.publish().is_err()));
        assert!(resolve_snapshot(root.path()).unwrap().generation.unwrap() > before);
    }
}
