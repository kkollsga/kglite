//! Where kglite spills its mmap-backed column blobs, and who reclaims the ones
//! a killed process left behind.
//!
//! Two producers, one root and one janitor:
//!
//! * a **load** with any column blob of 256 KB or more mints
//!   `<root>/kglite_portable_<pid>_<nanos_hex><seq_hex>/` and writes the blob
//!   there so the column can be mmap'd instead of heap-allocated. A load with
//!   no such blob — every small `.kgl` — creates nothing;
//! * a graph under a **`set_memory_limit`** mints
//!   `<root>/kglite_spill_<pid>_<nanos_hex><seq_hex>/` the first time
//!   `maybe_spill_columns` has to materialise a store to files.
//!
//! `<root>` is `KGLITE_TMPDIR` when it is set and non-empty, otherwise
//! [`std::env::temp_dir`] — for both, which is why both mint through this
//! module rather than reaching for `temp_dir()` themselves.
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
//! and [`memory_limit_temp_dir`] remove sibling directories, of **either**
//! prefix, whose embedded pid is dead. The sweep is deliberately timid — every
//! predicate below fails towards *keeping* a directory, because deleting one
//! that is still mapped corrupts a running graph while keeping one merely
//! defers a reclaim to the next process.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Prefix a `.kgl` load's spill directories carry.
const PORTABLE_PREFIX: &str = "kglite_portable_";

/// Prefix a `set_memory_limit` graph's spill directories carry.
const MEMORY_LIMIT_PREFIX: &str = "kglite_spill_";

/// Every prefix this module mints, and therefore the only names the sweep will
/// even consider.
///
/// The janitor is keyed on this list rather than on one constant because the
/// two producers leak identically — the drop that removes the tree is the only
/// cleanup either has, and a signal, an OOM kill or a panic-abort skips it —
/// so a prefix that mints without appearing here is an accumulation with no
/// gate, which is exactly what `kglite_spill_` was until this list existed.
const SPILL_PREFIXES: [&str; 2] = [PORTABLE_PREFIX, MEMORY_LIMIT_PREFIX];

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
///
/// Both producers resolve their root here. `maybe_spill_columns` used to call
/// `std::env::temp_dir()` directly, so an operator who pointed `KGLITE_TMPDIR`
/// at a roomy volume still had memory-limit spills land in `$TMPDIR`.
fn spill_root() -> PathBuf {
    match std::env::var_os("KGLITE_TMPDIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => std::env::temp_dir(),
    }
}

/// Distinguishes two spill directories minted inside one clock tick. The pid
/// separates processes and the clock orders the directories a process minted,
/// but the clock is not a unique value: `CLOCK_REALTIME` advances in ~41.7 ns
/// steps on arm64 macOS, so two threads loading `.kgl` files at once do read
/// the same nanosecond. Sharing a directory is a data race with `DirGraph`'s
/// drop — the first graph released removes the tree the second is still
/// filling — and it reproduced as `EEXIST`, `EINVAL` and `ENOENT` out of
/// `load_file` at roughly 1 in 600 concurrent loads before this counter
/// existed.
static NEXT_SPILL_DIR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The name of one spill directory, as a pure function of its inputs so the
/// uniqueness it promises is testable.
///
/// `seq` is fixed-width so that two names are equal only when both inputs are:
/// a variable-width counter could pair a different tick with a different
/// sequence and spell the same directory.
fn spill_dir_name(prefix: &str, pid: u32, nanos: u128, seq: u64) -> String {
    format!("{prefix}{pid}_{nanos:x}{seq:016x}")
}

/// A fresh, unique directory under `prefix`, after this process has taken its
/// one turn at reclaiming dead predecessors' directories.
///
/// The counter is shared across prefixes, which costs nothing and means the
/// two producers cannot collide with each other either.
fn mint_spill_dir(prefix: &str) -> PathBuf {
    sweep_once();
    spill_root().join(spill_dir_name(
        prefix,
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NEXT_SPILL_DIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ))
}

/// A fresh, unique spill directory for this load, after this process has taken
/// its one turn at reclaiming dead predecessors' directories.
///
/// The path is a *name*, not a directory: nothing is created here, and a load
/// whose columns all stay on the heap creates nothing at all (the spill writer
/// mints the tree on its first blob — see `ColumnStore::load_typed_vec`).
pub(super) fn portable_temp_dir() -> PathBuf {
    mint_spill_dir(PORTABLE_PREFIX)
}

/// A fresh, unique spill directory for a `set_memory_limit` graph — the
/// counterpart of [`portable_temp_dir`] for `DirGraph::maybe_spill_columns`,
/// which mints one lazily the first time it has to materialise a store.
///
/// Like the load's, this is a *name*: the caller creates the tree, registers it
/// on the graph, and the last `Arc` holder removes it.
///
/// **The sequence counter is load-bearing here for the same reason it is
/// there.** This name's only varying part was the wall clock, and
/// `CLOCK_REALTIME` advances in ~41.7 ns steps on arm64 macOS, so two graphs in
/// one process crossing their memory limit together read the same nanosecond,
/// share a directory, and the first one dropped runs `remove_dir_all` over
/// columns the second still has mapped.
pub(crate) fn memory_limit_temp_dir() -> PathBuf {
    mint_spill_dir(MEMORY_LIMIT_PREFIX)
}

/// The pid encoded in a spill-directory name, or `None` for any name that is
/// not exactly `<one of [`SPILL_PREFIXES`]><decimal pid>_<lowercase hex
/// suffix>`, where the suffix is the tick and sequence [`spill_dir_name`]
/// joins.
///
/// Strict on purpose: this predicate is the entire guard between the sweep and
/// whatever else shares the temp root, so anything it cannot fully account for
/// is somebody else's and is left alone.
fn parse_spill_pid(name: &str) -> Option<u32> {
    let rest = SPILL_PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))?;
    let (pid, suffix) = rest.split_once('_')?;
    if pid.is_empty() || !pid.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_hexdigit()) {
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

    /// Two loads that read the same clock tick must still get their own
    /// directory. The name's only varying part was the wall clock, and
    /// `CLOCK_REALTIME` is not a per-call unique value: on arm64 macOS it
    /// advances in ~41.7 ns steps, so two threads loading `.kgl` files at once
    /// can read the same nanosecond. Sharing a directory means the first graph
    /// dropped runs `remove_dir_all` over the second's spill files while it is
    /// still filling them — its next `create_dir_all`/`write` fails, or the
    /// column it already mapped is the last reference to a deleted file.
    #[test]
    fn two_spills_reading_one_clock_tick_get_different_directories() {
        let pid = std::process::id();
        let tick = 0x1234_5678_9abc_u128;
        // Both producers, because both mint from the clock and both hand the
        // directory to a drop that runs `remove_dir_all` over it.
        for prefix in SPILL_PREFIXES {
            let first = spill_dir_name(prefix, pid, tick, 0);
            let second = spill_dir_name(prefix, pid, tick, 1);
            assert_ne!(
                first, second,
                "two spills in one clock tick shared a directory ({prefix})"
            );
            // Both must stay sweepable: the janitor only reclaims names it can
            // fully parse (see `parse_spill_pid`).
            assert_eq!(parse_spill_pid(&first), Some(pid));
            assert_eq!(parse_spill_pid(&second), Some(pid));
        }
        // And the two producers cannot collide with each other.
        assert_ne!(
            spill_dir_name(PORTABLE_PREFIX, pid, tick, 0),
            spill_dir_name(MEMORY_LIMIT_PREFIX, pid, tick, 0)
        );
    }

    #[test]
    fn parses_only_the_exact_spill_name() {
        assert_eq!(parse_spill_pid("kglite_portable_1234_1a2b3c"), Some(1234));
        assert_eq!(parse_spill_pid("kglite_spill_1234_1a2b3c"), Some(1234));
        // The same malformations under both prefixes: the memory-limit half
        // inherits the whole predicate, not a looser cousin of it.
        for prefix in SPILL_PREFIXES {
            for suffix in [
                "1234",
                "1234_",
                "_1a2b",
                "1234_1a2b_3c",
                "12x4_1a2b",
                "+12_1a2b",
                "1234_zzz",
                "1234_+1a",
                "0_1a2b",
                "99999999999999999999_1a2b",
                "notapid_x",
            ] {
                let bad = format!("{prefix}{suffix}");
                assert_eq!(parse_spill_pid(&bad), None, "{bad} parsed as a spill dir");
            }
        }
        for bad in [
            "kglite-portable-1234-1a2b",
            "kglite-spill-1234-1a2b",
            "kglite_spilled_1234_1a2b",
            "someone-elses-data",
            ".DS_Store",
        ] {
            assert_eq!(parse_spill_pid(bad), None, "{bad} parsed as a spill dir");
        }
    }

    /// The whole predicate, run once per prefix so the memory-limit half is
    /// held to the identical contract rather than a looser one: only a
    /// directory that is old, dead-pid, well-named and ours-by-prefix goes.
    #[test]
    fn sweeps_old_dead_pid_dirs_only() {
        for prefix in SPILL_PREFIXES {
            let root = tempfile::tempdir().unwrap();
            let dead = dead_pid();
            let old_orphan = mint(
                root.path(),
                &format!("{prefix}{dead}_aa"),
                Duration::from_secs(7200),
            );
            let young_orphan = mint(
                root.path(),
                &format!("{prefix}{dead}_bb"),
                Duration::from_secs(60),
            );
            let live = mint(
                root.path(),
                &format!("{prefix}{}_cc", std::process::id()),
                Duration::from_secs(7200),
            );
            let junk = mint(root.path(), "not-a-spill-dir", Duration::from_secs(7200));
            let malformed = mint(
                root.path(),
                &format!("{prefix}{dead}"),
                Duration::from_secs(7200),
            );

            let outcome = sweep_orphans(root.path(), std::process::id(), SystemTime::now());

            assert_eq!(
                outcome,
                SweepOutcome {
                    removed: 1,
                    failed: 0
                },
                "{prefix}"
            );
            assert!(!old_orphan.exists(), "{prefix}");
            assert!(young_orphan.exists(), "{prefix}");
            assert!(live.exists(), "{prefix}");
            assert!(junk.exists(), "{prefix}");
            assert!(malformed.exists(), "{prefix}");
        }
    }

    /// One sweep reclaims both producers' leftovers. The regression this pins:
    /// `kglite_spill_` accumulated unswept from the day `set_memory_limit`
    /// shipped, because the janitor knew only the load's prefix.
    #[test]
    fn one_sweep_reclaims_both_producers_orphans() {
        let root = tempfile::tempdir().unwrap();
        let dead = dead_pid();
        let old = Duration::from_secs(7200);
        let from_a_load = mint(root.path(), &format!("{PORTABLE_PREFIX}{dead}_aa"), old);
        let from_a_limit = mint(root.path(), &format!("{MEMORY_LIMIT_PREFIX}{dead}_bb"), old);

        let outcome = sweep_orphans(root.path(), std::process::id(), SystemTime::now());

        assert_eq!(
            outcome,
            SweepOutcome {
                removed: 2,
                failed: 0
            }
        );
        assert!(!from_a_load.exists());
        assert!(!from_a_limit.exists());
    }

    #[test]
    fn spares_a_symlink_wearing_a_spill_name() {
        let root = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        fs::write(target.path().join("precious.txt"), b"keep me").unwrap();
        let link = root
            .path()
            .join(format!("{MEMORY_LIMIT_PREFIX}{}_aa", dead_pid()));
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
        // One of each prefix, so a failure on the load's leftovers cannot stop
        // the memory-limit ones being reclaimed.
        let blocked = mint(root.path(), &format!("{PORTABLE_PREFIX}{dead}_aa"), old);
        let removable = mint(root.path(), &format!("{MEMORY_LIMIT_PREFIX}{dead}_bb"), old);
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
        let ours: Vec<PathBuf> = SPILL_PREFIXES
            .iter()
            .map(|prefix| {
                mint(
                    root.path(),
                    &format!("{prefix}{}_aa", std::process::id()),
                    Duration::from_secs(86_400),
                )
            })
            .collect();

        let outcome = sweep_orphans(root.path(), std::process::id(), SystemTime::now());

        assert_eq!(outcome, SweepOutcome::default());
        assert!(ours.iter().all(|dir| dir.exists()));
    }

    #[test]
    fn kglite_tmpdir_overrides_the_default_root() {
        // `spill_root` reads the environment, which is process-global; this
        // test is the only place that writes `KGLITE_TMPDIR` in the lib
        // binary, and it restores it before returning.
        let root = tempfile::tempdir().unwrap();
        std::env::set_var("KGLITE_TMPDIR", root.path());
        assert_eq!(spill_root(), root.path());
        // Both producers, because both used to be able to disagree: the
        // memory-limit path resolved `std::env::temp_dir()` itself and ignored
        // the override entirely.
        assert!(portable_temp_dir().starts_with(root.path()));
        assert!(memory_limit_temp_dir().starts_with(root.path()));

        std::env::set_var("KGLITE_TMPDIR", "");
        assert_eq!(spill_root(), std::env::temp_dir());
        std::env::remove_var("KGLITE_TMPDIR");
        assert_eq!(spill_root(), std::env::temp_dir());
    }
}
