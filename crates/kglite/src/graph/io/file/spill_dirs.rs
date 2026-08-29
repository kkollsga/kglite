//! Where a portable `.kgl` load spills its mmap-backed column blobs, and who
//! reclaims the ones a killed process left behind.
//!
//! A load with any column section of 256 KB or more mints
//! `<root>/kglite_portable_<pid>_<nanos_hex>/` and writes the blob there so the
//! column can be mmap'd instead of heap-allocated. `<root>` is `KGLITE_TMPDIR`
//! when it is set and non-empty, otherwise [`std::env::temp_dir`].
//!
//! Ownership is drop-based — the paths are registered on the loaded `DirGraph`
//! and the last `Arc` holder removes them — which covers every clean exit and
//! nothing else. A process killed by a signal, an OOM kill or a panic-abort
//! never runs that drop, and the tree stays. The OS is not the backstop it
//! looks like: macOS never sweeps `$TMPDIR` for a live login session, and a
//! downstream measured 4,377 orphaned trees totalling 8.5 GB accumulate there
//! in a single day of load-and-kill cycles.
//!
//! So each process sweeps once, at its own first spill: [`portable_temp_dir`]
//! removes sibling directories whose embedded pid is dead. The sweep is
//! deliberately timid — every predicate below fails towards *keeping* a
//! directory, because deleting one that is still mapped corrupts a running
//! graph while keeping one merely defers a reclaim to the next process.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Prefix every spill directory carries, and the only names the sweep will
/// even consider.
const SPILL_PREFIX: &str = "kglite_portable_";

/// How old a dead-pid directory must be before the sweep will remove it.
///
/// Absorbs pid reuse: the kernel can hand a fresh process the pid of one that
/// died, so a directory minted seconds ago by a live loader can carry a pid
/// that `kill(2)` — asked about the *new* owner — happens to report as gone.
/// An hour is far longer than the window between minting a spill directory and
/// registering it on the graph.
const ORPHAN_MIN_AGE: Duration = Duration::from_secs(3600);

/// Root the spill directories are minted under: `KGLITE_TMPDIR` when set and
/// non-empty, otherwise [`std::env::temp_dir`].
fn spill_root() -> PathBuf {
    match std::env::var_os("KGLITE_TMPDIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => std::env::temp_dir(),
    }
}

/// A fresh, unique spill directory for this load, after this process has taken
/// its one turn at reclaiming dead predecessors' directories.
pub(super) fn portable_temp_dir() -> PathBuf {
    sweep_once();
    spill_root().join(format!(
        "kglite_portable_{}_{:x}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ))
}

/// The pid encoded in a spill-directory name, or `None` for any name that is
/// not exactly `kglite_portable_<decimal pid>_<lowercase hex nanos>`.
///
/// Strict on purpose: this predicate is the entire guard between the sweep and
/// whatever else shares the temp root, so anything it cannot fully account for
/// is somebody else's and is left alone.
fn parse_spill_pid(name: &str) -> Option<u32> {
    let rest = name.strip_prefix(SPILL_PREFIX)?;
    let (pid, nanos) = rest.split_once('_')?;
    if pid.is_empty() || !pid.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if nanos.is_empty() || !nanos.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    // A pid that does not fit `u32` was never written by `std::process::id()`,
    // and pid 0 addresses a process *group* in `kill(2)` rather than a
    // process — neither can be a spill directory of ours.
    match pid.parse::<u32>() {
        Ok(0) | Err(_) => None,
        Ok(pid) => Some(pid),
    }
}

/// Whether `pid` still names a running process.
///
/// Only `ESRCH` — "no such process" — counts as dead. `EPERM` (a process we
/// may not signal) and every other errno are read as alive, so an unexpected
/// kernel answer keeps the directory instead of removing a live one's.
#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return true;
    };
    // SAFETY: signal 0 sends nothing — `kill` only runs its existence and
    // permission checks. `parse_spill_pid` rejects 0, so `pid` is strictly
    // positive and this can address neither a process group nor every process.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// What one sweep did. Returned rather than logged so the tests can assert the
/// counts; a failed removal is counted and stepped over, never propagated —
/// the load that triggered the sweep must not fail over another process's
/// leftovers.
#[cfg(unix)]
#[derive(Debug, Default, PartialEq, Eq)]
struct SweepOutcome {
    removed: usize,
    failed: usize,
}

/// Whether `path`, named `name`, is a spill directory this process may remove:
/// a real directory (not a symlink), named exactly like a spill dir, belonging
/// to neither us nor any live process, and old enough to be past the pid-reuse
/// window.
#[cfg(unix)]
fn is_reclaimable(path: &Path, name: &str, self_pid: u32, now: SystemTime) -> bool {
    let Some(pid) = parse_spill_pid(name) else {
        return false;
    };
    // Our own directories are live by definition — including the one
    // `portable_temp_dir` is about to mint, whose name shares this pid.
    if pid == self_pid {
        return false;
    }
    // `symlink_metadata`, not `metadata`: a symlink named like a spill
    // directory must not put its target in reach of `remove_dir_all`.
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !meta.file_type().is_dir() {
        return false;
    }
    // An unreadable or future mtime yields no age, so the margin cannot be
    // shown to have passed.
    let Ok(age) = meta.modified().and_then(|m| {
        now.duration_since(m)
            .map_err(|_| std::io::Error::other("mtime is in the future"))
    }) else {
        return false;
    };
    if age < ORPHAN_MIN_AGE {
        return false;
    }
    !pid_is_alive(pid)
}

/// Remove every reclaimable spill directory directly under `root`.
#[cfg(unix)]
fn sweep_orphans(root: &Path, self_pid: u32, now: SystemTime) -> SweepOutcome {
    let mut outcome = SweepOutcome::default();
    let Ok(entries) = std::fs::read_dir(root) else {
        return outcome;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !is_reclaimable(&path, name, self_pid, now) {
            continue;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => outcome.removed += 1,
            Err(_) => outcome.failed += 1,
        }
    }
    outcome
}

/// Sweep the current spill root, at most once in this process.
///
/// Once-guarded because the cost is a `read_dir` plus one `kill(2)` per
/// candidate and the orphans it finds cannot reappear while we run — nothing
/// but a *dead* process's leftovers is ever eligible. A consequence worth
/// knowing when reading the tests: a later change to `KGLITE_TMPDIR` moves
/// where spills land but does not earn the new root a sweep.
#[cfg(unix)]
fn sweep_once() {
    static SWEEP: std::sync::Once = std::sync::Once::new();
    SWEEP.call_once(|| {
        let _ = sweep_orphans(&spill_root(), std::process::id(), SystemTime::now());
    });
}

/// No janitor off unix.
///
/// The sweep turns on a liveness probe, and the Windows equivalent —
/// `OpenProcess`/`GetExitCodeProcess` — has its own pid-reuse and
/// access-denied semantics that nothing here has been run against. Shipping an
/// unverified one would put `remove_dir_all` behind a guess; leaving it off
/// leaves Windows exactly where it was, with drop-based cleanup and no sweep.
#[cfg(not(unix))]
fn sweep_once() {}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;

    fn mint(root: &Path, name: &str, age: Duration) -> PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(dir.join("type_0")).unwrap();
        fs::write(dir.join("type_0").join("col.bin"), b"payload").unwrap();
        fs::File::open(&dir)
            .unwrap()
            .set_modified(SystemTime::now() - age)
            .unwrap();
        dir
    }

    /// A child that has exited and been reaped: dead for as long as the kernel
    /// has not wrapped its pid counter back onto it, which no constant can
    /// promise.
    fn dead_pid() -> u32 {
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "exit 0"])
            .spawn()
            .unwrap();
        let pid = child.id();
        child.wait().unwrap();
        pid
    }

    #[test]
    fn parses_only_the_exact_spill_name() {
        assert_eq!(parse_spill_pid("kglite_portable_1234_1a2b3c"), Some(1234));
        for bad in [
            "kglite_portable_1234",
            "kglite_portable_1234_",
            "kglite_portable__1a2b",
            "kglite_portable_1234_1a2b_3c",
            "kglite_portable_12x4_1a2b",
            "kglite_portable_+12_1a2b",
            "kglite_portable_1234_zzz",
            "kglite_portable_1234_+1a",
            "kglite_portable_0_1a2b",
            "kglite_portable_99999999999999999999_1a2b",
            "kglite_portable_notapid_x",
            "kglite-portable-1234-1a2b",
            "someone-elses-data",
            ".DS_Store",
        ] {
            assert_eq!(parse_spill_pid(bad), None, "{bad} parsed as a spill dir");
        }
    }

    #[test]
    fn sweeps_old_dead_pid_dirs_only() {
        let root = tempfile::tempdir().unwrap();
        let dead = dead_pid();
        let old_orphan = mint(
            root.path(),
            &format!("{SPILL_PREFIX}{dead}_aa"),
            Duration::from_secs(7200),
        );
        let young_orphan = mint(
            root.path(),
            &format!("{SPILL_PREFIX}{dead}_bb"),
            Duration::from_secs(60),
        );
        let live = mint(
            root.path(),
            &format!("{SPILL_PREFIX}{}_cc", std::process::id()),
            Duration::from_secs(7200),
        );
        let junk = mint(root.path(), "not-a-spill-dir", Duration::from_secs(7200));
        let malformed = mint(
            root.path(),
            &format!("{SPILL_PREFIX}{dead}"),
            Duration::from_secs(7200),
        );

        let outcome = sweep_orphans(root.path(), std::process::id(), SystemTime::now());

        assert_eq!(
            outcome,
            SweepOutcome {
                removed: 1,
                failed: 0
            }
        );
        assert!(!old_orphan.exists());
        assert!(young_orphan.exists());
        assert!(live.exists());
        assert!(junk.exists());
        assert!(malformed.exists());
    }

    #[test]
    fn spares_a_symlink_wearing_a_spill_name() {
        let root = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        fs::write(target.path().join("precious.txt"), b"keep me").unwrap();
        let link = root.path().join(format!("{SPILL_PREFIX}{}_aa", dead_pid()));
        std::os::unix::fs::symlink(target.path(), &link).unwrap();

        let outcome = sweep_orphans(root.path(), std::process::id(), SystemTime::now());

        assert_eq!(outcome, SweepOutcome::default());
        assert!(target.path().join("precious.txt").exists());
        assert!(link.symlink_metadata().is_ok());
    }

    #[test]
    fn a_failed_removal_does_not_stop_the_sweep() {
        let root = tempfile::tempdir().unwrap();
        let dead = dead_pid();
        let old = Duration::from_secs(7200);
        let blocked = mint(root.path(), &format!("{SPILL_PREFIX}{dead}_aa"), old);
        let removable = mint(root.path(), &format!("{SPILL_PREFIX}{dead}_bb"), old);
        // A read-only parent makes the unlink inside it fail; the entry itself
        // stays listable and its own mtime unchanged.
        let mut perms = fs::metadata(&blocked).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o500);
        fs::set_permissions(&blocked, perms).unwrap();

        let outcome = sweep_orphans(root.path(), std::process::id(), SystemTime::now());

        assert_eq!(
            outcome,
            SweepOutcome {
                removed: 1,
                failed: 1
            }
        );
        assert!(!removable.exists(), "the sweep stopped at the failure");
        let mut perms = fs::metadata(&blocked).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o700);
        fs::set_permissions(&blocked, perms).unwrap();
    }

    #[test]
    fn our_own_pid_is_never_swept_however_old_the_dir_looks() {
        let root = tempfile::tempdir().unwrap();
        let ours = mint(
            root.path(),
            &format!("{SPILL_PREFIX}{}_aa", std::process::id()),
            Duration::from_secs(86_400),
        );

        let outcome = sweep_orphans(root.path(), std::process::id(), SystemTime::now());

        assert_eq!(outcome, SweepOutcome::default());
        assert!(ours.exists());
    }

    #[test]
    fn kglite_tmpdir_overrides_the_default_root() {
        // `spill_root` reads the environment, which is process-global; this
        // test is the only place that writes `KGLITE_TMPDIR` in the lib
        // binary, and it restores it before returning.
        let root = tempfile::tempdir().unwrap();
        std::env::set_var("KGLITE_TMPDIR", root.path());
        assert_eq!(spill_root(), root.path());
        assert!(portable_temp_dir().starts_with(root.path()));

        std::env::set_var("KGLITE_TMPDIR", "");
        assert_eq!(spill_root(), std::env::temp_dir());
        std::env::remove_var("KGLITE_TMPDIR");
        assert_eq!(spill_root(), std::env::temp_dir());
    }
}
