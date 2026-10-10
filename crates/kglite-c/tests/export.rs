//! Open-format exports through the C ABI: the lossless CSV tree, RDF 1.2, and
//! `language_maps` on RDF import. A graph handle is built through a session,
//! saved, and loaded back, since a session consumes its handle.

use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};

use kglite_c::{
    kglite_cypher_result_free, kglite_cypher_result_rows_json, kglite_export_csv,
    kglite_free_string, kglite_graph_free, kglite_graph_new, kglite_load_file,
    kglite_session_execute_mut, kglite_session_execute_read, kglite_session_export_csv,
    kglite_session_free, kglite_session_new, kglite_session_save, KgliteCypherResult, KgliteGraph,
    KgliteSession, KgliteStatusCode,
};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kglite_c_export_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn c(text: &str) -> CString {
    CString::new(text).unwrap()
}

fn take(json: *const c_char) -> serde_json::Value {
    assert!(!json.is_null());
    let value = serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap());
    unsafe { kglite_free_string(json) };
    value.unwrap()
}

fn run(session: *mut KgliteSession, query: &str, write: bool) -> serde_json::Value {
    let query_c = c(query);
    let mut result: *mut KgliteCypherResult = std::ptr::null_mut();
    let mut error: *const c_char = std::ptr::null();
    let rc = unsafe {
        if write {
            kglite_session_execute_mut(
                session,
                query_c.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut error,
            )
        } else {
            kglite_session_execute_read(
                session,
                query_c.as_ptr(),
                std::ptr::null(),
                &mut result,
                &mut error,
            )
        }
    };
    assert_eq!(rc, KgliteStatusCode::Ok, "{query}");
    let rows = take(unsafe { kglite_cypher_result_rows_json(result) });
    unsafe { kglite_cypher_result_free(result) };
    rows
}

/// An HR graph (people in departments) as a graph handle.
fn hr_graph(dir: &Path) -> *mut KgliteGraph {
    let graph = kglite_graph_new();
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_session_new(graph, &mut session) },
        KgliteStatusCode::Ok
    );
    for statement in [
        "CREATE (:Person {id: 1, title: 'Ada', level: 7}), (:Person {id: 2, title: 'Bo', level: 3})",
        "CREATE (:Department {id: 10, title: 'Platform'})",
        "MATCH (p:Person {id: 1}), (d:Department {id: 10}) CREATE (p)-[:WORKS_IN {since: 2020}]->(d)",
    ] {
        run(session, statement, true);
    }
    let file = c(dir.join("hr.kgl").to_str().unwrap());
    assert_eq!(
        unsafe { kglite_session_save(session, file.as_ptr(), 0, std::ptr::null_mut()) },
        KgliteStatusCode::Ok
    );
    unsafe { kglite_session_free(session) };
    let mut loaded: *mut KgliteGraph = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_load_file(file.as_ptr(), &mut loaded, std::ptr::null_mut()) },
        KgliteStatusCode::Ok
    );
    loaded
}

#[test]
fn export_csv_writes_the_lossless_tree() {
    let dir = scratch("csv");
    let graph = hr_graph(&dir);
    let out = dir.join("tree");
    let out_c = c(out.to_str().unwrap());
    let mut summary: *const c_char = std::ptr::null();
    let mut error: *const c_char = std::ptr::null();
    let rc = unsafe { kglite_export_csv(graph, out_c.as_ptr(), &mut summary, &mut error) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert!(error.is_null());
    let summary = take(summary);
    assert_eq!(summary["nodes"]["Person"], 2);
    assert_eq!(summary["connections"]["WORKS_IN"], 1);
    assert!(out.join("blueprint.json").is_file());
    assert!(out.join("manifest.json").is_file());
    unsafe { kglite_graph_free(graph) };
}

#[test]
fn export_csv_refuses_an_empty_output_dir_and_writes_nothing() {
    let dir = scratch("csv_empty");
    let graph = hr_graph(&dir);
    let out_c = c("");
    let mut error: *const c_char = std::ptr::null();
    let rc = unsafe { kglite_export_csv(graph, out_c.as_ptr(), std::ptr::null_mut(), &mut error) };
    assert_ne!(rc, KgliteStatusCode::Ok);
    assert!(!error.is_null());
    let message = unsafe { CStr::from_ptr(error) }
        .to_str()
        .unwrap()
        .to_string();
    unsafe { kglite_free_string(error) };
    assert!(message.contains("must not be empty"), "{message}");
    // An empty path used to resolve to the working directory.
    assert!(!Path::new("blueprint.json").exists());
    unsafe { kglite_graph_free(graph) };
}

#[test]
fn export_csv_rejects_null_arguments() {
    let mut error: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_export_csv(
            std::ptr::null_mut(),
            c("x").as_ptr(),
            std::ptr::null_mut(),
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
}

#[cfg(feature = "rdf")]
mod rdf {
    use super::*;
    use kglite_c::{kglite_export_rdf, kglite_load_rdf, kglite_load_rdf_with_options};

    fn load_back(path: &Path) -> *mut KgliteGraph {
        let path_c = c(path.to_str().unwrap());
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut error: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_load_rdf(
                path_c.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                -1,
                &mut graph,
                std::ptr::null_mut(),
                &mut error,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        graph
    }

    fn rows(graph: *mut KgliteGraph, query: &str) -> serde_json::Value {
        let mut session: *mut KgliteSession = std::ptr::null_mut();
        assert_eq!(
            unsafe { kglite_session_new(graph, &mut session) },
            KgliteStatusCode::Ok
        );
        let rows = run(session, query, false);
        unsafe { kglite_session_free(session) };
        rows
    }

    #[test]
    fn export_rdf_round_trips_through_load_rdf() {
        let dir = scratch("rdf");
        let graph = hr_graph(&dir);
        for (name, format) in [
            ("g.nq", std::ptr::null()),
            ("g.out", c("trig").into_raw().cast_const()),
        ] {
            let path = dir.join(name);
            let path_c = c(path.to_str().unwrap());
            let mut summary: *const c_char = std::ptr::null();
            let mut error: *const c_char = std::ptr::null();
            let rc = unsafe {
                kglite_export_rdf(
                    graph,
                    path_c.as_ptr(),
                    format,
                    std::ptr::null(),
                    0,
                    &mut summary,
                    &mut error,
                )
            };
            assert_eq!(rc, KgliteStatusCode::Ok);
            assert!(take(summary)["statements"].as_u64().unwrap() > 0);
            if !format.is_null() {
                drop(unsafe { CString::from_raw(format.cast_mut().cast()) });
                // A `.out` file is not an RDF extension: copy to the TriG name to load.
                std::fs::copy(&path, dir.join("g.trig")).unwrap();
            }
        }
        for name in ["g.nq", "g.trig"] {
            let back = load_back(&dir.join(name));
            let people = rows(
                back,
                "MATCH (p:Person) RETURN p.id AS id, p.level AS level ORDER BY id",
            );
            assert_eq!(
                people,
                serde_json::json!([{"id": 1, "level": 7}, {"id": 2, "level": 3}])
            );
        }
        unsafe { kglite_graph_free(graph) };
    }

    #[test]
    fn export_rdf_reports_bad_format_and_base() {
        let dir = scratch("rdf_bad");
        let graph = hr_graph(&dir);
        let path_c = c(dir.join("g.nq").to_str().unwrap());
        for (format, base, expected) in [
            (Some("turtle"), None, "Unknown RDF format"),
            (None, Some("http://schema.org/"), "well-known"),
        ] {
            let format_c = format.map(c);
            let base_c = base.map(c);
            let mut error: *const c_char = std::ptr::null();
            let rc = unsafe {
                kglite_export_rdf(
                    graph,
                    path_c.as_ptr(),
                    format_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                    base_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                    0,
                    std::ptr::null_mut(),
                    &mut error,
                )
            };
            assert_eq!(rc, KgliteStatusCode::InvalidArgument);
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_string();
            unsafe { kglite_free_string(error) };
            assert!(message.contains(expected), "{message}");
        }
        unsafe { kglite_graph_free(graph) };
    }

    #[test]
    fn load_rdf_with_options_reads_language_maps() {
        let dir = scratch("langs");
        let path = dir.join("names.nq");
        std::fs::write(
            &path,
            "<http://ex.org/a> <http://ex.org/name> \"Platform\"@en .\n\
             <http://ex.org/a> <http://ex.org/name> \"Plattform\"@de .\n",
        )
        .unwrap();
        let path_c = c(path.to_str().unwrap());
        let mut names = Vec::new();
        for language_maps in [0u8, 1] {
            let mut graph: *mut KgliteGraph = std::ptr::null_mut();
            let mut error: *const c_char = std::ptr::null();
            let rc = unsafe {
                kglite_load_rdf_with_options(
                    path_c.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    -1,
                    language_maps,
                    &mut graph,
                    std::ptr::null_mut(),
                    &mut error,
                )
            };
            assert_eq!(rc, KgliteStatusCode::Ok);
            names.push(rows(graph, "MATCH (n) RETURN properties(n) AS p"));
        }
        // Off: the tags are dropped and the values fold to a list; on: a map by tag.
        assert_eq!(
            names[0][0]["p"]["name"],
            serde_json::json!(["Platform", "Plattform"])
        );
        assert_eq!(
            names[1][0]["p"]["name"],
            serde_json::json!({"en": "Platform", "de": "Plattform"})
        );
    }
}

/// A live session over the HR rows, for the session-scoped exports.
fn hr_session() -> *mut KgliteSession {
    let mut session: *mut KgliteSession = std::ptr::null_mut();
    assert_eq!(
        unsafe { kglite_session_new(kglite_graph_new(), &mut session) },
        KgliteStatusCode::Ok
    );
    run(
        session,
        "CREATE (:Person {id: 1, title: 'Ada'}), (:Department {id: 10, title: 'Platform'})",
        true,
    );
    run(
        session,
        "MATCH (p:Person {id: 1}), (d:Department {id: 10}) CREATE (p)-[:WORKS_IN]->(d)",
        true,
    );
    session
}

#[test]
fn session_export_csv_writes_the_committed_state() {
    let dir = scratch("session_csv");
    let session = hr_session();
    let out_c = c(dir.join("tree").to_str().unwrap());
    let mut summary: *const c_char = std::ptr::null();
    let mut error: *const c_char = std::ptr::null();
    let rc =
        unsafe { kglite_session_export_csv(session, out_c.as_ptr(), &mut summary, &mut error) };
    assert_eq!(rc, KgliteStatusCode::Ok);
    let summary = take(summary);
    assert_eq!(summary["nodes"]["Person"], 1);
    assert_eq!(summary["connections"]["WORKS_IN"], 1);
    assert!(dir.join("tree").join("manifest.json").is_file());
    // The session is borrowed, not consumed.
    let rows = run(session, "MATCH (n) RETURN count(n) AS c", false);
    assert!(rows.to_string().contains('2'), "{rows}");
    unsafe { kglite_session_free(session) };
}

#[test]
fn session_export_csv_rejects_null_arguments() {
    let mut error: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_session_export_csv(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::NullPointer);
}

#[cfg(feature = "rdf")]
#[test]
fn session_export_rdf_round_trips_through_load_rdf() {
    let dir = scratch("session_rdf");
    let session = hr_session();
    let out = dir.join("g.nq");
    let out_c = c(out.to_str().unwrap());
    let mut summary: *const c_char = std::ptr::null();
    let mut error: *const c_char = std::ptr::null();
    let rc = unsafe {
        kglite_c::kglite_session_export_rdf(
            session,
            out_c.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            &mut summary,
            &mut error,
        )
    };
    assert_eq!(rc, KgliteStatusCode::Ok);
    assert_eq!(take(summary)["nodes"]["Person"], 1);
    assert!(out.is_file());
    unsafe { kglite_session_free(session) };
}
