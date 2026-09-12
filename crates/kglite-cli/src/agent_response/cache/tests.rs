use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};
use tempfile::TempDir;

use super::*;

#[derive(Clone)]
struct FakeClock(Arc<AtomicU64>);

impl Clock for FakeClock {
    fn unix_seconds(&self) -> Result<u64> {
        Ok(self.0.load(Ordering::SeqCst))
    }
}

#[derive(Clone)]
struct SequenceIds(Arc<Mutex<VecDeque<u128>>>);

impl SequenceIds {
    fn ascending() -> Self {
        Self(Arc::new(Mutex::new((1..=256).collect())))
    }

    fn exact(values: impl IntoIterator<Item = u128>) -> Self {
        Self(Arc::new(Mutex::new(values.into_iter().collect())))
    }
}

impl IdSource for SequenceIds {
    fn next_id(&self) -> Result<[u8; 16]> {
        self.0
            .lock()
            .unwrap()
            .pop_front()
            .map(u128::to_be_bytes)
            .ok_or_else(|| anyhow!("test ID sequence exhausted"))
    }
}

fn private_root() -> (TempDir, PathBuf) {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("cache");
    (parent, root)
}

fn fixture(
    root: &Path,
    now: u64,
    limits: Limits,
) -> (ResultCache<FakeClock, SequenceIds>, Arc<AtomicU64>) {
    let clock = Arc::new(AtomicU64::new(now));
    (
        ResultCache::with_parts(
            root.to_path_buf(),
            limits,
            FakeClock(clock.clone()),
            SequenceIds::ascending(),
        ),
        clock,
    )
}

fn load_value<C: Clock, I: IdSource>(cache: &ResultCache<C, I>, handle: &str) -> Option<Value> {
    cache.load(handle).unwrap().map(|loaded| loaded.envelope)
}

fn entry_paths(root: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(root.join("entries"))
        .unwrap()
        .map(|item| item.unwrap().path())
        .collect();
    paths.sort();
    paths
}

fn total_entry_bytes(root: &Path) -> u64 {
    entry_paths(root)
        .iter()
        .map(|path| fs::symlink_metadata(path).unwrap().len())
        .sum()
}

#[test]
fn store_and_handle_only_reopen_preserve_exact_envelope_and_provenance() {
    let (_parent, root) = private_root();
    let (cache, _) = fixture(&root, 10, Limits::default());
    let envelope = json!({
        "schema_version": 1,
        "kind": "cypher_result",
        "columns": ["x", "x"],
        "rows": [[1, null]],
        "diagnostics": {"warnings": []},
        "coverage": {},
        "identity": {"graph": "gone"},
        "operation": {"kind": "read"},
        "representation": {"format": "agent"}
    });
    let handle = cache.store("workspace-a", envelope.clone()).unwrap();
    drop(cache);

    // A separate caller has only the user cache root and opaque handle. It
    // does not reconstruct the original cwd, namespace, graph, or query.
    let (reopened, _) = fixture(&root, 11, Limits::default());
    let loaded = reopened.load(&handle).unwrap().unwrap();
    assert_eq!(loaded.namespace, "workspace-a");
    assert_eq!(loaded.created_at_unix_seconds, 10);
    assert_eq!(loaded.envelope, envelope);
}

#[test]
fn ttl_boundary_expires_and_physically_removes_entry_on_access() {
    let (_parent, root) = private_root();
    let limits = Limits {
        ttl: Duration::from_secs(10),
        ..Limits::default()
    };
    let (cache, clock) = fixture(&root, 50, limits);
    let handle = cache.store("a", json!({"x": 1})).unwrap();
    clock.store(59, Ordering::SeqCst);
    assert_eq!(load_value(&cache, &handle), Some(json!({"x": 1})));
    clock.store(60, Ordering::SeqCst);
    assert_eq!(load_value(&cache, &handle), None);
    assert!(entry_paths(&root).is_empty());
}

#[test]
fn entry_cap_is_global_across_abandoned_namespaces() {
    let (_parent, root) = private_root();
    let limits = Limits {
        max_entries: 2,
        max_bytes: 1_000_000,
        ..Limits::default()
    };
    let (cache, clock) = fixture(&root, 10, limits);
    let first = cache.store("workspace-a", json!({"n": 1})).unwrap();
    clock.store(11, Ordering::SeqCst);
    let second = cache.store("workspace-b", json!({"n": 2})).unwrap();
    clock.store(12, Ordering::SeqCst);
    let third = cache.store("workspace-c", json!({"n": 3})).unwrap();
    assert_eq!(load_value(&cache, &first), None);
    assert_eq!(load_value(&cache, &second), Some(json!({"n": 2})));
    assert_eq!(load_value(&cache, &third), Some(json!({"n": 3})));
    assert_eq!(entry_paths(&root).len(), 2);
}

#[test]
fn same_timestamp_evicts_creation_order_not_random_handle_order() {
    let (_parent, root) = private_root();
    let clock = Arc::new(AtomicU64::new(10));
    let cache = ResultCache::with_parts(
        root,
        Limits {
            max_entries: 2,
            max_bytes: 1_000_000,
            ..Limits::default()
        },
        FakeClock(clock),
        SequenceIds::exact([100, 99, 98]),
    );
    let first = cache.store("a", json!({"order": 1})).unwrap();
    let second = cache.store("a", json!({"order": 2})).unwrap();
    let third = cache.store("a", json!({"order": 3})).unwrap();
    assert_eq!(load_value(&cache, &first), None);
    assert_eq!(load_value(&cache, &second), Some(json!({"order": 2})));
    assert_eq!(load_value(&cache, &third), Some(json!({"order": 3})));
}

#[test]
fn byte_cap_uses_complete_serialized_record_sizes() {
    let (_calibration_parent, calibration_root) = private_root();
    let (calibration, _) = fixture(&calibration_root, 10, Limits::default());
    calibration
        .store("a", json!({"blob": "a".repeat(31)}))
        .unwrap();
    calibration
        .store("a", json!({"blob": "b".repeat(37)}))
        .unwrap();
    let exact_two = total_entry_bytes(&calibration_root);

    let (_parent, root) = private_root();
    let limits = Limits {
        max_entries: 32,
        max_bytes: exact_two,
        ..Limits::default()
    };
    let (cache, _) = fixture(&root, 10, limits);
    let first = cache.store("a", json!({"blob": "a".repeat(31)})).unwrap();
    let second = cache.store("a", json!({"blob": "b".repeat(37)})).unwrap();
    assert!(total_entry_bytes(&root) <= exact_two);
    let third = cache.store("a", json!({"blob": "c".repeat(41)})).unwrap();
    assert!(total_entry_bytes(&root) <= exact_two);
    assert_eq!(load_value(&cache, &first), None);
    assert!(load_value(&cache, &second).is_some() || load_value(&cache, &third).is_some());
}

#[test]
fn one_record_larger_than_global_byte_cap_is_rejected_without_eviction() {
    let (_parent, root) = private_root();
    let limits = Limits {
        max_entries: 32,
        max_bytes: 300,
        ..Limits::default()
    };
    let (cache, _) = fixture(&root, 10, limits);
    let survivor = cache.store("a", json!({"small": true})).unwrap();
    let before = fs::read(cache.entry_path(&survivor)).unwrap();
    let error = cache
        .store("a", json!({"blob": "x".repeat(1_000)}))
        .unwrap_err();
    assert!(error.to_string().contains("exceeds"));
    assert_eq!(fs::read(cache.entry_path(&survivor)).unwrap(), before);
    assert_eq!(entry_paths(&root).len(), 1);
}

#[test]
fn stale_regular_and_symlink_temps_are_removed_without_following_target() {
    let (_parent, root) = private_root();
    let (cache, _) = fixture(&root, 10, Limits::default());
    let survivor = cache.store("a", json!({"ok": true})).unwrap();
    let regular = cache.entries_dir().join(".orphan.tmp-1");
    fs::write(&regular, b"partial").unwrap();
    set_private_file(&regular);

    #[cfg(unix)]
    {
        let external = root.parent().unwrap().join("external-sentinel");
        fs::write(&external, b"keep").unwrap();
        std::os::unix::fs::symlink(&external, cache.entries_dir().join(".link.tmp-2")).unwrap();
        assert_eq!(load_value(&cache, &survivor), Some(json!({"ok": true})));
        assert_eq!(fs::read(&external).unwrap(), b"keep");
    }
    #[cfg(not(unix))]
    assert_eq!(load_value(&cache, &survivor), Some(json!({"ok": true})));

    assert!(!regular.exists());
    assert_eq!(entry_paths(&root).len(), 1);
}

#[test]
fn duplicate_random_ids_retry_without_clobbering_prior_record() {
    let (_parent, root) = private_root();
    let cache = ResultCache::with_parts(
        root,
        Limits::default(),
        FakeClock(Arc::new(AtomicU64::new(10))),
        SequenceIds::exact([1, 1, 2]),
    );
    let first = cache.store("a", json!({"n": 1})).unwrap();
    let second = cache.store("a", json!({"n": 2})).unwrap();
    assert_ne!(first, second);
    assert_eq!(load_value(&cache, &first), Some(json!({"n": 1})));
    assert_eq!(load_value(&cache, &second), Some(json!({"n": 2})));
}

#[test]
fn malformed_handles_never_reach_path_lookup() {
    let (_parent, root) = private_root();
    let (cache, _) = fixture(&root, 10, Limits::default());
    for handle in [
        "../../victim",
        "kgr_",
        "kgr_ABCDEF00000000000000000000000000",
        "kgr_00/0000000000000000000000000000",
        "kgr_00000000000000000000000000000g",
    ] {
        assert!(cache.load(handle).is_err(), "{handle}");
    }
    assert!(
        !root.exists(),
        "validation happens before cache-root creation"
    );
}

#[test]
fn corrupt_and_non_record_files_are_cleaned_on_access() {
    let (_parent, root) = private_root();
    let (cache, _) = fixture(&root, 10, Limits::default());
    let handle = cache.store("a", json!({"ok": true})).unwrap();
    let corrupt = cache.entries_dir().join(format!(
        "{HANDLE_PREFIX}000000000000000000000000000000ff.json"
    ));
    fs::write(&corrupt, b"not json").unwrap();
    set_private_file(&corrupt);
    let junk = cache.entries_dir().join("junk");
    fs::write(&junk, b"junk").unwrap();
    set_private_file(&junk);
    assert!(cache.load(&handle).unwrap().is_some());
    assert!(!corrupt.exists());
    assert!(!junk.exists());
}

#[test]
fn partial_write_and_rename_failures_leave_no_candidate_or_temp() {
    for fault in [
        TestFault::BeforeWrite,
        TestFault::AfterPrefix,
        TestFault::BeforeRename,
    ] {
        let (_parent, root) = private_root();
        let (mut cache, _) = fixture(&root, 10, Limits::default());
        cache.fault = fault;
        assert!(cache.store("a", json!({"candidate": true})).is_err());
        assert!(entry_paths(&root).is_empty(), "fault {fault:?}");
    }
}

#[test]
fn failed_publication_at_capacity_may_evict_oldest_but_stays_within_caps() {
    let (_parent, root) = private_root();
    let limits = Limits {
        max_entries: 1,
        max_bytes: 1_000_000,
        ..Limits::default()
    };
    let (mut cache, _) = fixture(&root, 10, limits);
    let old = cache.store("old", json!({"old": true})).unwrap();
    cache.fault = TestFault::BeforeRename;
    assert!(cache.store("new", json!({"new": true})).is_err());
    assert_eq!(load_value(&cache, &old), None);
    assert!(entry_paths(&root).is_empty());
    assert!(total_entry_bytes(&root) <= limits.max_bytes);
}

#[test]
fn concurrent_writers_publish_unique_complete_records_under_global_cap() {
    let (_parent, root) = private_root();
    let cache = ResultCache::new(root.clone());
    let mut workers = Vec::new();
    for index in 0..16_u64 {
        let cache = cache.clone();
        workers.push(thread::spawn(move || {
            let value = json!({"index": index, "blob": "x".repeat(1_000)});
            let handle = cache
                .store(&format!("workspace-{}", index % 4), value.clone())
                .unwrap();
            (handle, value)
        }));
    }
    let completed: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    let mut handles: Vec<_> = completed.iter().map(|(handle, _)| handle.clone()).collect();
    handles.sort();
    handles.dedup();
    assert_eq!(handles.len(), 16);
    for (handle, expected) in completed {
        assert_eq!(cache.load(&handle).unwrap().unwrap().envelope, expected);
    }
    assert_eq!(entry_paths(&root).len(), 16);
    assert!(total_entry_bytes(&root) <= DEFAULT_MAX_BYTES);
}

#[test]
fn concurrent_processes_publish_records_loadable_by_root_and_handle() {
    let (_parent, root) = private_root();
    let output_dir = root.parent().unwrap().join("process-output");
    fs::create_dir(&output_dir).unwrap();
    let executable = std::env::current_exe().unwrap();
    let mut children = Vec::new();
    for index in 0..8_u64 {
        let output = output_dir.join(index.to_string());
        children.push((
            index,
            output.clone(),
            Command::new(&executable)
                .arg("cache_process_writer_child")
                .arg("--nocapture")
                .env("KGLITE_CACHE_CHILD_ROOT", &root)
                .env("KGLITE_CACHE_CHILD_OUTPUT", output)
                .env("KGLITE_CACHE_CHILD_INDEX", index.to_string())
                .spawn()
                .unwrap(),
        ));
    }

    let cache = ResultCache::new(root);
    for (index, output, mut child) in children {
        assert!(child.wait().unwrap().success());
        let handle = fs::read_to_string(output).unwrap();
        let loaded = cache.load(&handle).unwrap().unwrap();
        assert_eq!(loaded.namespace, format!("process-{index}"));
        assert_eq!(loaded.envelope, json!({"index": index}));
    }
}

// The parent test invokes this same test binary with a name filter. A normal
// suite run leaves the environment unset, so this fixture returns immediately.
#[test]
fn cache_process_writer_child() {
    let Some(root) = std::env::var_os("KGLITE_CACHE_CHILD_ROOT") else {
        return;
    };
    let output = PathBuf::from(std::env::var_os("KGLITE_CACHE_CHILD_OUTPUT").unwrap());
    let index: u64 = std::env::var("KGLITE_CACHE_CHILD_INDEX")
        .unwrap()
        .parse()
        .unwrap();
    let cache = ResultCache::new(PathBuf::from(root));
    let handle = cache
        .store(&format!("process-{index}"), json!({"index": index}))
        .unwrap();
    fs::write(output, handle).unwrap();
}

#[test]
fn purge_all_removes_every_namespace_and_preserves_lock_owner() {
    let (_parent, root) = private_root();
    let (cache, _) = fixture(&root, 10, Limits::default());
    cache.store("a", json!({"a": 1})).unwrap();
    cache.store("b", json!({"b": 2})).unwrap();
    cache.purge_all().unwrap();
    assert!(entry_paths(&root).is_empty());
    assert!(root.join("lock").is_file());
}

#[cfg(unix)]
#[test]
fn rejects_symlink_root_public_root_and_nonregular_entry_without_deleting_targets() {
    use std::os::unix::fs::{symlink, PermissionsExt};

    let parent = tempfile::tempdir().unwrap();
    let real = parent.path().join("real");
    fs::create_dir(&real).unwrap();
    fs::set_permissions(&real, fs::Permissions::from_mode(0o700)).unwrap();
    let linked = parent.path().join("linked");
    symlink(&real, &linked).unwrap();
    assert!(ResultCache::new(linked).store("a", json!({})).is_err());

    let public = parent.path().join("public");
    fs::create_dir(&public).unwrap();
    fs::set_permissions(&public, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(ResultCache::new(public).store("a", json!({})).is_err());

    let root = parent.path().join("cache");
    let cache = ResultCache::new(root.clone());
    let handle = cache.store("a", json!({"safe": true})).unwrap();
    let bad = cache.entries_dir().join("directory.json");
    fs::create_dir(&bad).unwrap();
    assert!(cache.load(&handle).is_err());
    assert!(bad.is_dir());
}

#[cfg(unix)]
fn set_private_file(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

#[cfg(not(unix))]
fn set_private_file(_path: &Path) {}
