//! `kglite_open_session` and the verbs on its handle: durable open at each
//! level, crash recovery, lease contention, real read-only, the lifecycle
//! calls and the row-limit execute variants.

use kglite_c::*;
use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};

struct TestDir(PathBuf);

impl TestDir {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("kglite-c-durable-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn graph(&self) -> PathBuf {
        self.0.join("g.kgl")
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn take(p: *const c_char) -> Option<String> {
    (!p.is_null()).then(|| {
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
        unsafe { kglite_free_string(p) };
        s
    })
}

struct Opened {
    status: KgliteStatusCode,
    session: *mut KgliteSession,
    info: Option<serde_json::Value>,
    error: Option<String>,
}

fn open(path: &Path, options: serde_json::Value) -> Opened {
    let path_c = cstr(path.to_str().unwrap());
    let options_c = cstr(&options.to_string());
    let mut session = std::ptr::null_mut();
    let mut info: *const c_char = std::ptr::null();
    let mut error: *const c_char = std::ptr::null();
    let status = unsafe {
        kglite_open_session(
            path_c.as_ptr(),
            options_c.as_ptr(),
            &mut session,
            &mut info,
            &mut error,
        )
    };
    Opened {
        status,
        session,
        info: take(info).map(|j| serde_json::from_str(&j).unwrap()),
        error: take(error),
    }
}

fn open_ok(path: &Path, options: serde_json::Value) -> Opened {
    let opened = open(path, options);
    assert_eq!(opened.status, KgliteStatusCode::Ok, "{:?}", opened.error);
    opened
}

fn write(session: *mut KgliteSession, query: &str) -> KgliteStatusCode {
    let q = cstr(query);
    let mut result = std::ptr::null_mut();
    let mut error: *const c_char = std::ptr::null();
    let status = unsafe {
        kglite_session_execute_mut(
            session,
            q.as_ptr(),
            std::ptr::null(),
            &mut result,
            &mut error,
        )
    };
    unsafe { kglite_cypher_result_free(result) };
    take(error);
    status
}

fn rows(session: *const KgliteSession, query: &str) -> serde_json::Value {
    let q = cstr(query);
    let mut result = std::ptr::null_mut();
    let mut error: *const c_char = std::ptr::null();
    let status = unsafe {
        kglite_session_execute_read(
            session,
            q.as_ptr(),
            std::ptr::null(),
            &mut result,
            &mut error,
        )
    };
    assert_eq!(status, KgliteStatusCode::Ok, "{query}: {:?}", take(error));
    let json = take(unsafe { kglite_cypher_result_rows_json(result) }).unwrap();
    unsafe { kglite_cypher_result_free(result) };
    serde_json::from_str(&json).unwrap()
}

fn count(session: *const KgliteSession) -> i64 {
    rows(session, "MATCH (n:T) RETURN count(n) AS c")[0]["c"]
        .as_i64()
        .unwrap()
}

fn close(session: *mut KgliteSession) -> KgliteStatusCode {
    let mut error: *const c_char = std::ptr::null();
    let status = unsafe { kglite_session_close(session, &mut error) };
    take(error);
    status
}

fn free(session: *mut KgliteSession) {
    unsafe { kglite_session_free(session) };
}

fn create(durability: &str) -> serde_json::Value {
    serde_json::json!({"durability": durability, "create_if_missing": true})
}

#[test]
fn committed_writes_survive_free_without_close_at_full_and_normal() {
    for level in ["full", "normal"] {
        let dir = TestDir::new(&format!("recover-{level}"));
        let path = dir.graph();
        let first = open_ok(&path, create(level));
        assert_eq!(first.info.as_ref().unwrap()["durability"], level);
        assert_eq!(first.info.as_ref().unwrap()["created"], true);
        assert_eq!(
            write(first.session, "CREATE (:T {id: 1})"),
            KgliteStatusCode::Ok
        );
        assert_eq!(
            write(first.session, "CREATE (:T {id: 2})"),
            KgliteStatusCode::Ok
        );
        // Freed, never closed and never checkpointed: only the log holds them.
        free(first.session);

        let second = open_ok(&path, serde_json::json!({"durability": level}));
        assert_eq!(
            count(second.session),
            2,
            "level {level}: log replayed on open"
        );
        free(second.session);
    }
}

#[test]
fn without_a_log_an_unclosed_session_loses_its_writes() {
    // The contrast that makes the recovery test non-vacuous: the same
    // sequence at durability "off" leaves nothing to replay.
    let dir = TestDir::new("off-loses");
    let path = dir.graph();
    let first = open_ok(&path, create("off"));
    assert_eq!(
        write(first.session, "CREATE (:T {id: 1})"),
        KgliteStatusCode::Ok
    );
    free(first.session);
    let second = open(&path, serde_json::json!({}));
    match second.status {
        KgliteStatusCode::FileNotFound => {}
        KgliteStatusCode::Ok => {
            assert_eq!(count(second.session), 0, "off must not recover");
            free(second.session);
        }
        other => panic!("unexpected {other:?}: {:?}", second.error),
    }
}

#[test]
fn close_checkpoints_so_even_level_off_persists() {
    let dir = TestDir::new("close-off");
    let path = dir.graph();
    let first = open_ok(&path, create("off"));
    assert_eq!(
        write(first.session, "CREATE (:T {id: 1})"),
        KgliteStatusCode::Ok
    );
    assert_eq!(close(first.session), KgliteStatusCode::Ok);
    assert_eq!(close(first.session), KgliteStatusCode::Ok, "idempotent");
    free(first.session);
    let second = open_ok(&path, serde_json::json!({"durability": "off"}));
    assert_eq!(count(second.session), 1);
    free(second.session);
}

#[test]
fn lease_contention_is_status_102_with_holder_details_and_close_releases_it() {
    let dir = TestDir::new("lease");
    let path = dir.graph();
    let holder = open_ok(&path, create("full"));
    let refused = open(&path, serde_json::json!({}));
    assert_eq!(refused.status, KgliteStatusCode::WriterLeaseHeld);
    assert!(refused.session.is_null());
    assert!(refused
        .error
        .unwrap()
        .contains(&std::process::id().to_string()));
    let details: serde_json::Value =
        serde_json::from_str(&take(kglite_last_error_details_json()).unwrap()).unwrap();
    assert_eq!(details["code"], "WriterLeaseHeld");
    assert_eq!(details["pid"], std::process::id());
    assert_eq!(details["self"], true);
    assert!(details["since"].is_string(), "{details}");

    // close releases the lease while the handle is still allocated.
    assert_eq!(close(holder.session), KgliteStatusCode::Ok);
    let successor = open_ok(&path, serde_json::json!({}));
    free(successor.session);
    free(holder.session);
}

#[test]
fn free_releases_the_lease() {
    let dir = TestDir::new("lease-free");
    let path = dir.graph();
    let first = open_ok(&path, create("full"));
    free(first.session);
    free(open_ok(&path, serde_json::json!({})).session);
}

#[test]
fn lock_timeout_minus_one_is_a_real_read_only_open() {
    let dir = TestDir::new("readonly");
    let path = dir.graph();

    // Missing path: not created, no sidecar, no lease file.
    let missing = open(&path, serde_json::json!({"lock_timeout_ms": -1}));
    assert_eq!(missing.status, KgliteStatusCode::FileNotFound);
    assert!(missing.session.is_null());
    assert_eq!(
        std::fs::read_dir(&dir.0).unwrap().count(),
        0,
        "nothing created"
    );

    // create_if_missing and storage are refused outright.
    for refused in [
        serde_json::json!({"lock_timeout_ms": -1, "create_if_missing": true}),
        serde_json::json!({"lock_timeout_ms": -1, "storage": "memory"}),
        serde_json::json!({"lock_timeout_ms": -1, "durability": "full"}),
    ] {
        assert_eq!(
            open(&path, refused).status,
            KgliteStatusCode::InvalidArgument
        );
    }
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);

    let writer = open_ok(&path, create("full"));
    assert_eq!(
        write(writer.session, "CREATE (:T {id: 1})"),
        KgliteStatusCode::Ok
    );
    assert_eq!(close(writer.session), KgliteStatusCode::Ok);
    free(writer.session);
    let on_disk = std::fs::read(&path).unwrap();

    // A live writer excludes other writers but not a reader.
    let live = open_ok(&path, serde_json::json!({}));
    let reader = open_ok(&path, serde_json::json!({"lock_timeout_ms": -1}));
    let info = reader.info.unwrap();
    assert_eq!(info["read_only"], true);
    assert_eq!(info["durability"], "off");
    assert_eq!(count(reader.session), 1);
    assert_eq!(
        write(reader.session, "CREATE (:T {id: 2})"),
        KgliteStatusCode::ReadOnly
    );
    let mut error: *const c_char = std::ptr::null();
    assert_eq!(
        unsafe {
            kglite_session_checkpoint(
                reader.session,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut error,
            )
        },
        KgliteStatusCode::ReadOnly
    );
    take(error);
    assert_eq!(close(reader.session), KgliteStatusCode::Ok);
    free(reader.session);
    free(live.session);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        on_disk,
        "read-only wrote nothing"
    );
}

#[test]
fn valid_time_default_option_is_applied() {
    let dir = TestDir::new("valid-time");
    let path = dir.graph();
    let seed = open_ok(&path, create("full"));
    for statement in [
        "CREATE (:Well {id: 1, vf: date('2000-01-01'), vt: date('2010-01-01')}), \
         (:Well {id: 2, vf: date('2005-01-01')})",
        "CALL db.temporal.declare({node: 'Well', from: 'vf', to: 'vt', convention: 'closed'}) \
         YIELD declared RETURN declared",
    ] {
        assert_eq!(write(seed.session, statement), KgliteStatusCode::Ok);
    }
    assert_eq!(close(seed.session), KgliteStatusCode::Ok);
    free(seed.session);

    let query = "MATCH (w:Well) RETURN w.id AS id ORDER BY id";
    let today = open_ok(&path, serde_json::json!({"valid_time_default": "today"}));
    assert_eq!(rows(today.session, query), serde_json::json!([{"id": 2}]));
    free(today.session);
    let all = open_ok(&path, serde_json::json!({"valid_time_default": "all"}));
    assert_eq!(
        rows(all.session, query),
        serde_json::json!([{"id": 1}, {"id": 2}])
    );
    free(all.session);
    let dated = open_ok(
        &path,
        serde_json::json!({"valid_time_default": "2003-06-30"}),
    );
    assert_eq!(rows(dated.session, query), serde_json::json!([{"id": 1}]));
    free(dated.session);
    // The read-only open honours it too.
    let reader = open_ok(
        &path,
        serde_json::json!({"lock_timeout_ms": -1, "valid_time_default": "all"}),
    );
    assert_eq!(
        rows(reader.session, query),
        serde_json::json!([{"id": 1}, {"id": 2}])
    );
    free(reader.session);
}

#[test]
fn bad_options_are_rejected() {
    let dir = TestDir::new("badopts");
    let path = dir.graph();
    for bad in [
        serde_json::json!({"durablity": "full"}),
        serde_json::json!({"durability": "sometimes"}),
        serde_json::json!({"storage": "tape"}),
        serde_json::json!({"lock_timeout_ms": -2}),
        serde_json::json!({"valid_time_default": "tomorrowish"}),
        serde_json::json!({"auto_checkpoint_wal_mib": -1}),
        serde_json::json!({"auto_checkpoint_wal_mib": "16"}),
        serde_json::json!([1]),
    ] {
        let opened = open(&path, bad.clone());
        assert_eq!(opened.status, KgliteStatusCode::InvalidArgument, "{bad}");
    }
    // Missing path without the flag is FILE_NOT_FOUND, and nothing is created.
    assert_eq!(
        open(&path, serde_json::json!({})).status,
        KgliteStatusCode::FileNotFound
    );
    assert!(!path.exists());
}

#[test]
fn sync_and_checkpoint_semantics() {
    let dir = TestDir::new("sync");
    let path = dir.graph();
    let normal = open_ok(&path, create("normal"));
    let mut error: *const c_char = std::ptr::null();
    assert_eq!(
        write(normal.session, "CREATE (:T {id: 1})"),
        KgliteStatusCode::Ok
    );
    assert_eq!(
        unsafe { kglite_session_sync(normal.session, &mut error) },
        KgliteStatusCode::Ok
    );

    let (mut written, mut version) = (9u8, 0u64);
    let checkpoint = |s: *mut KgliteSession, w: &mut u8, v: &mut u64| {
        let mut error: *const c_char = std::ptr::null();
        let status = unsafe { kglite_session_checkpoint(s, w, v, &mut error) };
        take(error);
        status
    };
    assert_eq!(
        checkpoint(normal.session, &mut written, &mut version),
        KgliteStatusCode::Ok
    );
    assert_eq!(written, 1, "first checkpoint always writes");
    let first_version = version;
    assert_eq!(
        checkpoint(normal.session, &mut written, &mut version),
        KgliteStatusCode::Ok
    );
    assert_eq!(
        (written, version),
        (0, first_version),
        "unchanged -> skipped"
    );
    assert_eq!(
        write(normal.session, "CREATE (:T {id: 2})"),
        KgliteStatusCode::Ok
    );
    assert_eq!(
        checkpoint(normal.session, &mut written, &mut version),
        KgliteStatusCode::Ok
    );
    assert_eq!(written, 1);
    assert!(version > first_version);
    free(normal.session);

    // No log, no sync.
    let off = open_ok(&path, serde_json::json!({"durability": "off"}));
    assert_eq!(
        unsafe { kglite_session_sync(off.session, &mut error) },
        KgliteStatusCode::NotDurable
    );
    take(error);
    free(off.session);

    // A session built from a graph handle has no path to checkpoint to.
    let graph = kglite_graph_new();
    let mut plain = std::ptr::null_mut();
    unsafe { kglite_session_new(graph, &mut plain) };
    assert_eq!(
        checkpoint(plain, &mut written, &mut version),
        KgliteStatusCode::InvalidArgument
    );
    assert_eq!(
        close(plain),
        KgliteStatusCode::Ok,
        "close on a plain session is a no-op"
    );
    free(plain);
}

#[test]
fn closed_session_refuses_writes() {
    let dir = TestDir::new("closed");
    let first = open_ok(&dir.graph(), create("full"));
    assert_eq!(close(first.session), KgliteStatusCode::Ok);
    assert_eq!(
        write(first.session, "CREATE (:T {id: 1})"),
        KgliteStatusCode::InvalidArgument
    );
    free(first.session);
}

#[test]
fn batches_and_edges_on_a_durable_session_are_logged() {
    let dir = TestDir::new("batch");
    let path = dir.graph();
    let first = open_ok(&path, create("full"));
    let batch = cstr(
        &serde_json::json!([
            {"query": "CREATE (:T {id: 1})"},
            {"query": "CREATE (:T {id: 2})"},
        ])
        .to_string(),
    );
    let (mut out, mut error): (*const c_char, *const c_char) = (std::ptr::null(), std::ptr::null());
    assert_eq!(
        unsafe {
            kglite_session_execute_mut_batch(first.session, batch.as_ptr(), &mut out, &mut error)
        },
        KgliteStatusCode::Ok,
        "{:?}",
        take(error)
    );
    take(out);
    let edges = cstr(
        &serde_json::json!([{"src_id": 1, "src_type": "T", "dst_id": 2, "dst_type": "T", "type": "R"}])
            .to_string(),
    );
    assert_eq!(
        unsafe { kglite_create_edges_batch(first.session, edges.as_ptr(), &mut out, &mut error) },
        KgliteStatusCode::Ok,
        "{:?}",
        take(error)
    );
    take(out);
    // The log keeps accepting commits after the batch paths (no divergence latch).
    assert_eq!(
        write(first.session, "CREATE (:T {id: 3})"),
        KgliteStatusCode::Ok
    );
    free(first.session);

    let second = open_ok(&path, serde_json::json!({}));
    assert_eq!(count(second.session), 3);
    assert_eq!(
        rows(
            second.session,
            "MATCH (:T)-[r:R]->(:T) RETURN count(r) AS c"
        )[0]["c"],
        1
    );
    free(second.session);
}

#[test]
fn log_bypassing_writers_are_refused_on_a_durable_session() {
    let dir = TestDir::new("bypass");
    let first = open_ok(&dir.graph(), create("full"));
    assert_eq!(
        write(first.session, "CREATE (:T {id: 1})"),
        KgliteStatusCode::Ok
    );
    let (node_type, property) = (cstr("T"), cstr("id"));
    let (mut report, mut error): (*const c_char, *const c_char) =
        (std::ptr::null(), std::ptr::null());
    assert_eq!(
        unsafe {
            kglite_session_build_text_index(
                first.session,
                node_type.as_ptr(),
                property.as_ptr(),
                &mut report,
                &mut error,
            )
        },
        KgliteStatusCode::DurabilityFailed
    );
    take(error);
    // Refusal is a pre-check: the log still accepts commits.
    assert_eq!(
        write(first.session, "CREATE (:T {id: 2})"),
        KgliteStatusCode::Ok
    );
    free(first.session);
}

fn execute_ex(
    session: *mut KgliteSession,
    query: &str,
    options: Option<&KgliteExecuteOptions>,
    mutating: bool,
) -> (
    KgliteStatusCode,
    Option<serde_json::Value>,
    Option<serde_json::Value>,
) {
    let q = cstr(query);
    let options_ptr = options.map_or(std::ptr::null(), |o| o as *const _);
    let mut result = std::ptr::null_mut();
    let mut error: *const c_char = std::ptr::null();
    let status = unsafe {
        if mutating {
            kglite_session_execute_mut_ex(
                session,
                q.as_ptr(),
                std::ptr::null(),
                options_ptr,
                &mut result,
                &mut error,
            )
        } else {
            kglite_session_execute_read_ex(
                session,
                q.as_ptr(),
                std::ptr::null(),
                options_ptr,
                &mut result,
                &mut error,
            )
        }
    };
    take(error);
    if result.is_null() {
        return (status, None, None);
    }
    let rows =
        serde_json::from_str(&take(unsafe { kglite_cypher_result_rows_json(result) }).unwrap())
            .unwrap();
    let diagnostics = serde_json::from_str(
        &take(unsafe { kglite_cypher_result_diagnostics_json(result) }).unwrap(),
    )
    .unwrap();
    unsafe { kglite_cypher_result_free(result) };
    (status, Some(rows), Some(diagnostics))
}

fn options(row_limit: u64) -> KgliteExecuteOptions {
    KgliteExecuteOptions {
        struct_size: std::mem::size_of::<KgliteExecuteOptions>(),
        timeout_ms: 0,
        max_work_units: 0,
        row_limit,
        flags: 1,
        reserved: 0,
        cancel: std::ptr::null(),
    }
}

#[test]
fn row_limit_truncates_and_reports() {
    let dir = TestDir::new("rowlimit");
    let opened = open_ok(&dir.graph(), create("off"));
    let s = opened.session;

    let (status, rows, diagnostics) = execute_ex(
        s,
        "UNWIND range(1, 10) AS i RETURN i",
        Some(&options(3)),
        false,
    );
    assert_eq!(status, KgliteStatusCode::Ok);
    assert_eq!(rows.unwrap().as_array().unwrap().len(), 3);
    let diagnostics = diagnostics.unwrap();
    assert_eq!(diagnostics["row_limit"], 3);
    assert_eq!(diagnostics["total_rows"], 10);

    // Without the flag the field is ignored; a null block means no limits.
    let mut unflagged = options(3);
    unflagged.flags = 0;
    let (_, rows, _) = execute_ex(
        s,
        "UNWIND range(1, 10) AS i RETURN i",
        Some(&unflagged),
        false,
    );
    assert_eq!(rows.unwrap().as_array().unwrap().len(), 10);
    let (_, rows, _) = execute_ex(s, "UNWIND range(1, 10) AS i RETURN i", None, false);
    assert_eq!(rows.unwrap().as_array().unwrap().len(), 10);

    // A mutation caps what it reports, never what it writes.
    let (status, rows, diagnostics) = execute_ex(
        s,
        "UNWIND range(1, 5) AS i CREATE (n:T {id: i}) RETURN n.id AS id",
        Some(&options(2)),
        true,
    );
    assert_eq!(status, KgliteStatusCode::Ok);
    assert_eq!(rows.unwrap().as_array().unwrap().len(), 2);
    assert_eq!(diagnostics.unwrap()["total_rows"], 5);
    assert_eq!(count(s), 5);

    // A block too small to hold struct_size is refused.
    let mut tiny = options(1);
    tiny.struct_size = 1;
    let (status, rows, _) = execute_ex(s, "RETURN 1 AS x", Some(&tiny), false);
    assert_eq!(status, KgliteStatusCode::InvalidArgument);
    assert!(rows.is_none());

    // A shorter (older) block leaves later fields at zero: no row limit.
    let mut older = options(1);
    older.struct_size = std::mem::offset_of!(KgliteExecuteOptions, row_limit);
    let (_, rows, _) = execute_ex(s, "UNWIND range(1, 4) AS i RETURN i", Some(&older), false);
    assert_eq!(rows.unwrap().as_array().unwrap().len(), 4);
    free(s);
}

#[test]
fn mut_ex_on_a_durable_session_is_logged_and_refused_when_read_only() {
    let dir = TestDir::new("mutex");
    let path = dir.graph();
    let first = open_ok(&path, create("full"));
    let (status, _, _) = execute_ex(
        first.session,
        "CREATE (:T {id: 1}) RETURN 1 AS x",
        Some(&options(5)),
        true,
    );
    assert_eq!(status, KgliteStatusCode::Ok);
    assert_eq!(close(first.session), KgliteStatusCode::Ok);
    free(first.session);
    let reader = open_ok(&path, serde_json::json!({"lock_timeout_ms": -1}));
    let (status, _, _) = execute_ex(reader.session, "CREATE (:T {id: 9})", None, true);
    assert_eq!(status, KgliteStatusCode::ReadOnly);
    free(reader.session);
}

fn wal_len(path: &Path) -> u64 {
    let mut wal = path.as_os_str().to_owned();
    wal.push("-wal");
    std::fs::metadata(wal).map_or(0, |m| m.len())
}

/// Past `auto_checkpoint_wal_mib` the commit that crosses it folds the log
/// into the checkpoint inline; with `0` the log only grows. The bound is whole
/// MiB, so the writes carry a 64 KiB payload to cross 1 MiB in a few dozen
/// commits.
#[test]
fn auto_checkpoint_bounds_the_log_and_zero_disables_it() {
    let pad = "x".repeat(64 * 1024);
    for (mib, bounded) in [(1u64, true), (0, false)] {
        let dir = TestDir::new(&format!("autockpt-{mib}"));
        let path = dir.graph();
        let first = open_ok(
            &path,
            serde_json::json!({
                "durability": "normal", "create_if_missing": true,
                "auto_checkpoint_wal_mib": mib,
            }),
        );
        for id in 0..40 {
            let query = format!("CREATE (:T {{id: {id}, pad: '{pad}'}})");
            assert_eq!(write(first.session, &query), KgliteStatusCode::Ok);
        }
        let log = wal_len(&path);
        if bounded {
            assert!(path.exists(), "the checkpoint was written inline");
            assert!(log < 2 << 20, "log trimmed along the way, is {log} bytes");
        } else {
            assert!(!path.exists(), "no checkpoint when disabled");
            assert!(log > 2 << 20, "log grows unchecked, is {log} bytes");
        }
        // Freed without close: the checkpoint plus the log hold all 40.
        free(first.session);
        let second = open_ok(&path, serde_json::json!({"durability": "normal"}));
        assert_eq!(count(second.session), 40, "mib={mib}");
        free(second.session);
    }
}

/// Concurrent writers on one durable handle must all commit. A durable write
/// runs on a fork and commits optimistically; before the session write gate,
/// readers holding snapshots pushed writers onto that path and racing commits
/// came back `TransactionConflict` (the 0.19.6 Java shootout: 1 in ~7k).
#[test]
fn concurrent_durable_writers_with_readers_all_commit() {
    let dir = TestDir::new("concurrent-writers");
    let opened = open_ok(&dir.graph(), create("normal"));
    let session = opened.session as usize;
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let done = done.clone();
            std::thread::spawn(move || {
                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    count(session as *const KgliteSession);
                }
            })
        })
        .collect();
    let writers: Vec<_> = (0..8)
        .map(|w| {
            std::thread::spawn(move || {
                (0..50)
                    .map(|i| {
                        write(
                            session as *mut KgliteSession,
                            &format!("CREATE (:T {{w: {w}, i: {i}}})"),
                        )
                    })
                    .filter(|status| *status != KgliteStatusCode::Ok)
                    .count()
            })
        })
        .collect();
    let failed: usize = writers.into_iter().map(|w| w.join().unwrap()).sum();
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    for reader in readers {
        reader.join().unwrap();
    }
    assert_eq!(failed, 0, "writes refused under concurrency");
    assert_eq!(count(opened.session), 400);
    assert_eq!(close(opened.session), KgliteStatusCode::Ok);
    free(opened.session);
}
