use std::io;
use std::path::{Path, PathBuf};

use super::{read_sidecar, unreplayed, DurableOpenError};
use crate::graph::wal::{wal_path, DurabilityLevel, SyncMode, Wal};

/// Compare publication destinations without resolving the final component.
/// Atomic save replaces a final symlink or hardlink rather than its referent;
/// parent-directory aliases, relative paths and `.` still name the same target.
pub fn same_checkpoint_path(left: &Path, right: &Path) -> io::Result<bool> {
    fn destination(path: &Path) -> io::Result<PathBuf> {
        let name = path.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "checkpoint path needs a filename",
            )
        })?;
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        Ok(parent
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()?
            .join(name))
    }
    Ok(destination(left)? == destination(right)?)
}

/// Prepare an independent save destination while its writer lease is held.
///
/// Its log must be contained in its own checkpoint, not merely below an
/// unrelated source's LSN. Clear only proven-contained residue before replacing
/// that checkpoint, so a failed save leaves the old destination recoverable.
/// The returned log is empty and ready for the source owner's next LSN. The
/// caller publishes its path/log/lease together only after the save succeeds;
/// the original source log must remain untouched.
///
/// Callers opting out of writer leases must coordinate destination writers
/// themselves, as with an unlocked open. At `Off`, no log is created, but
/// existing contained frames are cleared so they cannot replay over the save.
pub fn prepare_save_as_target(
    checkpoint_path: &Path,
    level: DurabilityLevel,
) -> Result<Option<Wal>, DurableOpenError> {
    let wpath = wal_path(checkpoint_path);
    let frames = read_sidecar(&wpath)?;
    if !frames.is_empty() {
        let checkpoint_lsn = match crate::graph::io::file::checkpoint_lsn_from_file(checkpoint_path)
        {
            Ok(lsn) => lsn,
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(DurableOpenError::Io(error.to_string())),
        };
        if unreplayed(&frames, checkpoint_lsn) {
            return Err(DurableOpenError::Refused(format!(
                "the write-ahead log at '{}' holds commits its destination checkpoint \
                 does not contain; open that destination durably to recover them before save-as",
                wpath.display()
            )));
        }
    }
    if !level.logs() && frames.is_empty() {
        return Ok(None);
    }
    let mut wal = Wal::open(wpath, level.sync_mode().unwrap_or(SyncMode::Barrier))
        .map_err(|error| DurableOpenError::Io(error.to_string()))?;
    wal.reset()
        .map_err(|error| DurableOpenError::Io(error.to_string()))?;
    Ok(level.logs().then_some(wal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::io::file::save_graph;
    use crate::graph::wal::{recover, WalFrame};
    use std::sync::Arc;

    fn checkpoint(path: &Path, lsn: u64) {
        let mut graph = Arc::new(DirGraph::new());
        Arc::make_mut(&mut graph).checkpoint_lsn = lsn;
        save_graph(&mut graph, path.to_str().unwrap()).unwrap();
    }

    fn frame(path: &Path, lsn: u64) {
        Wal::open(wal_path(path), SyncMode::Barrier)
            .unwrap()
            .append(&WalFrame { lsn, ops: vec![] })
            .unwrap();
    }

    #[test]
    fn contained_residue_is_cleared_before_foreign_checkpoint() {
        for level in [
            DurabilityLevel::Off,
            DurabilityLevel::Normal,
            DurabilityLevel::Full,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("graph.kgl");
            checkpoint(&path, 9);
            frame(&path, 9);
            let wal = prepare_save_as_target(&path, level).unwrap();
            assert_eq!(wal.is_some(), level.logs());
            assert!(recover(&wal_path(&path)).unwrap().is_empty());
            assert_eq!(
                crate::graph::io::file::checkpoint_lsn_from_file(&path).unwrap(),
                9
            );
        }
    }

    #[test]
    fn missing_or_corrupt_checkpoint_never_discards_pending_frames() {
        for contents in [None, Some(b"bad checkpoint".as_slice())] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("graph.kgl");
            if let Some(bytes) = contents {
                std::fs::write(&path, bytes).unwrap();
            }
            frame(&path, 1);
            assert!(prepare_save_as_target(&path, DurabilityLevel::Full).is_err());
            assert_eq!(recover(&wal_path(&path)).unwrap()[0].lsn, 1);
        }
    }

    #[test]
    fn preparation_reads_metadata_without_loading_graph_sections() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("graph.kgl");
        checkpoint(&path, 3);
        let bytes = std::fs::read(&path).unwrap();
        let len = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        std::fs::write(&path, &bytes[..13 + len]).unwrap();
        frame(&path, 3);
        assert!(prepare_save_as_target(&path, DurabilityLevel::Full).is_ok());
    }

    #[test]
    fn empty_destination_off_does_not_create_a_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("graph.kgl");
        assert!(prepare_save_as_target(&path, DurabilityLevel::Off)
            .unwrap()
            .is_none());
        assert!(!wal_path(&path).exists());
        assert!(same_checkpoint_path(&path, &temp.path().join("./graph.kgl")).unwrap());
    }
}
