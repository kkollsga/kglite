use kglite::api::durable::{recover, wal_path, SyncMode, Wal, WalFrame};
use kglite::api::io::{save_graph, GraphWriterLease};
use kglite::api::DirGraph;
use kglite_c::*;
use std::ffi::{CStr, CString};
use std::sync::Arc;

struct TestDirectory(std::path::PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn foreign_checkpoint_stamp_cannot_discard_target_recovery_through_c() {
    for route in ["graph", "durable_graph", "session"] {
        let temp = std::env::temp_dir().join(format!(
            "kglite-c-save-recovery-{}-{route}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp).unwrap();
        let _cleanup = TestDirectory(temp.clone());
        let source = temp.join("source.kgl");
        let target = temp.join("target.kgl");
        let mut graph = Arc::new(DirGraph::new());
        Arc::make_mut(&mut graph).checkpoint_lsn = 5;
        save_graph(&mut graph, source.to_str().unwrap()).unwrap();
        Arc::make_mut(&mut graph).checkpoint_lsn = 1;
        save_graph(&mut graph, target.to_str().unwrap()).unwrap();
        Wal::open(wal_path(&target), SyncMode::Barrier)
            .unwrap()
            .append(&WalFrame {
                lsn: 2,
                ops: vec![],
            })
            .unwrap();
        let original = std::fs::read(&target).unwrap();
        let _lease = GraphWriterLease::acquire(&target, std::time::Duration::ZERO).unwrap();
        let source_c = CString::new(source.to_str().unwrap()).unwrap();
        let target_c = CString::new(target.to_str().unwrap()).unwrap();
        let mut handle = std::ptr::null_mut();
        let mut error = std::ptr::null();
        // All handles and strings remain live across these synchronous C calls.
        unsafe {
            assert_eq!(
                kglite_load_file(source_c.as_ptr(), &mut handle, &mut error),
                KgliteStatusCode::Ok
            );
            let status = match route {
                "graph" => kglite_save_graph(handle, target_c.as_ptr(), &mut error),
                "durable_graph" => {
                    kglite_save_graph_durable(handle, target_c.as_ptr(), 1, &mut error)
                }
                _ => {
                    let mut session = std::ptr::null_mut();
                    assert_eq!(
                        kglite_session_new(handle, &mut session),
                        KgliteStatusCode::Ok
                    );
                    let status = kglite_session_save(session, target_c.as_ptr(), 1, &mut error);
                    kglite_session_free(session);
                    status
                }
            };
            assert_ne!(status, KgliteStatusCode::Ok, "{route}");
            assert!(!error.is_null());
            assert!(CStr::from_ptr(error)
                .to_string_lossy()
                .contains("write-ahead"));
            kglite_free_string(error);
            if route != "session" {
                kglite_graph_free(handle);
            }
        }
        assert_eq!(std::fs::read(&target).unwrap(), original);
        assert_eq!(recover(&wal_path(&target)).unwrap()[0].lsn, 2);
        drop(_lease);
        std::fs::remove_dir_all(&temp).unwrap();
    }
}
