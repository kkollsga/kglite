//! Shared graph open-or-create lifecycle used by server-style bindings.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use fs2::FileExt;

use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::load_file;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

/// How [`open_or_create_graph`] obtained the returned graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenDisposition {
    Opened,
    Created,
}

/// Graph plus the lifecycle decision made while opening it.
pub struct OpenGraphResult {
    pub graph: Arc<DirGraph>,
    pub disposition: OpenDisposition,
    /// Identity verified stable across an existing-path load, or captured
    /// immediately after creating a new path-backed graph.
    pub identity: GraphFileIdentity,
}

/// Cross-process writer ownership for a graph path. The sibling lock file is
/// persistent; OS advisory-lock teardown, not PID-file deletion, owns liveness.
pub struct GraphWriterLease {
    file: File,
}

impl GraphWriterLease {
    pub fn acquire(graph_path: &Path, timeout: Duration) -> io::Result<Self> {
        let path = writer_lease_path(graph_path);
        let started = Instant::now();
        loop {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            match file.try_lock_exclusive() {
                Ok(()) => {
                    file.set_len(0)?;
                    file.rewind()?;
                    writeln!(file, "pid={}", std::process::id())?;
                    file.sync_data()?;
                    return Ok(Self { file });
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    if started.elapsed() >= timeout {
                        return Err(io::Error::new(
                            io::ErrorKind::WouldBlock,
                            format!(
                                "timed out waiting for writer lease {}; another writer is active",
                                path.display()
                            ),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(error) => return Err(error),
            }
        }
    }
}

impl Drop for GraphWriterLease {
    fn drop(&mut self) {
        // Closing a locked descriptor normally releases its advisory lock,
        // but doing so explicitly gives every fs2 backend the same teardown
        // boundary and lets another writer acquire immediately after drop.
        let _ = FileExt::unlock(&self.file);
    }
}

fn writer_lease_path(graph_path: &Path) -> std::path::PathBuf {
    let mut lock = graph_path.as_os_str().to_os_string();
    lock.push(".lock");
    lock.into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetadataIdentity {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    handle: Arc<same_file::Handle>,
}

impl MetadataIdentity {
    fn capture(path: &Path) -> io::Result<(Self, std::fs::Metadata)> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(not(windows))]
        let metadata = std::fs::metadata(path)?;
        #[cfg(windows)]
        let (metadata, handle) = {
            let handle = Arc::new(same_file::Handle::from_path(path)?);
            let metadata = handle.as_file().metadata()?;
            (metadata, handle)
        };

        Ok((
            Self {
                len: metadata.len(),
                modified: metadata.modified()?,
                #[cfg(unix)]
                device: metadata.dev(),
                #[cfg(unix)]
                inode: metadata.ino(),
                #[cfg(windows)]
                handle,
            },
            metadata,
        ))
    }

    fn open_snapshot(&self, _path: &Path) -> io::Result<File> {
        #[cfg(windows)]
        return self.handle.as_file().try_clone();
        #[cfg(not(windows))]
        File::open(_path)
    }
}

/// Identity of a graph path at load/save time. Disk directories include the
/// published `CURRENT` pointer bytes, so a generation promotion is detected
/// even when the root directory inode itself is unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFileIdentity {
    root: Option<MetadataIdentity>,
    current: Option<(MetadataIdentity, Vec<u8>)>,
}

impl GraphFileIdentity {
    pub fn capture(path: &Path) -> io::Result<Self> {
        let (root, metadata) = match MetadataIdentity::capture(path) {
            Ok(captured) => captured,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    root: None,
                    current: None,
                });
            }
            Err(error) => return Err(error),
        };
        if !metadata.is_dir() {
            return Ok(Self {
                root: Some(root),
                current: None,
            });
        }

        let current_path = path.join("CURRENT");
        let (current_identity, current_metadata) = match MetadataIdentity::capture(&current_path) {
            Ok(captured) => captured,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    root: Some(root),
                    current: None,
                });
            }
            Err(error) => return Err(error),
        };
        if current_metadata.len() > 4096 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "disk graph CURRENT pointer exceeds 4096 bytes",
            ));
        }
        let mut bytes = Vec::with_capacity(current_metadata.len() as usize);
        current_identity
            .open_snapshot(&current_path)?
            .read_to_end(&mut bytes)?;
        Ok(Self {
            root: Some(root),
            current: Some((current_identity, bytes)),
        })
    }
}

/// Open an existing graph, or create an empty graph in `create_mode` when the
/// path is absent.
///
/// Existing graphs always auto-detect their persisted storage mode. Passing
/// `None` deliberately makes a missing path an error, which lets command-line
/// bindings reject typos unless the operator explicitly opts into creation.
///
/// This function makes a lifecycle decision, not a write-ownership promise.
/// A caller that may later publish to `path` must hold its own cross-process
/// writer lease across the read/modify/save interval. Read-only callers should
/// not acquire such a lease merely to open a graph.
pub fn open_or_create_graph(
    path: &Path,
    create_mode: Option<StorageMode>,
) -> io::Result<OpenGraphResult> {
    match std::fs::metadata(path) {
        Ok(_) => {
            let before = GraphFileIdentity::capture(path)?;
            let graph = load_file(&path.to_string_lossy())?;
            let identity = GraphFileIdentity::capture(path)?;
            if identity != before {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!("graph path {} changed while it was loading", path.display()),
                ));
            }
            return Ok(OpenGraphResult {
                graph,
                disposition: OpenDisposition::Opened,
                identity,
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
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
    Ok(OpenGraphResult {
        graph: Arc::new(graph),
        disposition: OpenDisposition::Created,
        identity: GraphFileIdentity::capture(path)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::GraphRead;
    use std::process::Command;

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
        assert_eq!(memory.disposition, OpenDisposition::Created);
        assert!(!memory.graph.graph.is_mapped());
        assert!(!memory.graph.graph.is_disk());

        let disk_path = tmp.path().join("disk");
        let disk = open_or_create_graph(&disk_path, Some(StorageMode::Disk)).unwrap();
        assert_eq!(disk.disposition, OpenDisposition::Created);
        assert!(disk.graph.graph.is_disk());
        assert!(disk_path.is_dir());
    }

    #[test]
    fn existing_graph_is_loaded_regardless_of_create_mode() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("existing.kgl");
        let mut graph = Arc::new(DirGraph::new());
        crate::graph::io::file::save_graph(&mut graph, &path.to_string_lossy()).unwrap();

        let loaded = open_or_create_graph(&path, Some(StorageMode::Disk)).unwrap();
        assert_eq!(loaded.disposition, OpenDisposition::Opened);
        assert!(!loaded.graph.graph.is_disk());
    }

    #[test]
    fn disk_identity_tracks_current_generation_content() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("CURRENT"), b"gen_00000000000000000001\n").unwrap();
        let first = GraphFileIdentity::capture(tmp.path()).unwrap();
        std::fs::write(tmp.path().join("CURRENT"), b"gen_00000000000000000002\n").unwrap();
        let second = GraphFileIdentity::capture(tmp.path()).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn writer_lease_child() {
        let Some(graph_path) = std::env::var_os("KGLITE_LEASE_CHILD_GRAPH") else {
            return;
        };
        let ready = std::env::var_os("KGLITE_LEASE_CHILD_READY").unwrap();
        let _lease = GraphWriterLease::acquire(Path::new(&graph_path), Duration::ZERO).unwrap();
        std::fs::write(ready, b"ready").unwrap();
        std::thread::sleep(Duration::from_secs(60));
    }

    #[test]
    fn crashed_process_releases_writer_lease() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("graph.kgl");
        let ready = tmp.path().join("ready");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "graph::io::open::tests::writer_lease_child",
                "--nocapture",
            ])
            .env("KGLITE_LEASE_CHILD_GRAPH", &graph)
            .env("KGLITE_LEASE_CHILD_READY", &ready)
            .spawn()
            .unwrap();
        let started = Instant::now();
        while !ready.exists() && started.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(ready.exists(), "child did not acquire lease");
        assert!(GraphWriterLease::acquire(&graph, Duration::ZERO).is_err());
        child.kill().unwrap();
        child.wait().unwrap();
        GraphWriterLease::acquire(&graph, Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn writer_lease_serializes_open_create_and_publish() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("barrier.kgl");
        {
            let _lease = GraphWriterLease::acquire(&path, Duration::ZERO).unwrap();
            let mut created = open_or_create_graph(&path, Some(StorageMode::Memory)).unwrap();
            assert_eq!(created.disposition, OpenDisposition::Created);
            crate::graph::io::file::save_graph(&mut created.graph, &path.to_string_lossy())
                .unwrap();
        }
        let _lease = GraphWriterLease::acquire(&path, Duration::ZERO).unwrap();
        let opened = open_or_create_graph(&path, Some(StorageMode::Memory)).unwrap();
        assert_eq!(opened.disposition, OpenDisposition::Opened);
    }

    #[test]
    fn dropping_lease_does_not_delete_replacement_path() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("replacement.kgl");
        let lock = writer_lease_path(&graph);
        let moved = tmp.path().join("moved.lock");
        let lease = GraphWriterLease::acquire(&graph, Duration::ZERO).unwrap();
        std::fs::rename(&lock, &moved).unwrap();
        std::fs::write(&lock, b"replacement\n").unwrap();
        drop(lease);
        assert_eq!(std::fs::read(&lock).unwrap(), b"replacement\n");
    }
}
