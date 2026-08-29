//! `KGLITE_MAX_LOAD_MB` — the load-memory ceiling's environment half.
//!
//! Its own integration binary, and its scenarios run in *child* processes, for
//! two reasons the option half (`file_load_ceiling_tests`) does not have:
//! the variable is read once per `LoadOptions::new()` and the environment is
//! process-global, and the unparseable-value warning goes to this process's
//! stderr, which can only be read from outside it.
//!
//! Shape: each `#[test]` here re-runs *this binary* with `--exact
//! child_scenario --ignored`, handing the case name and the fixture path
//! through the environment, then asserts on the child's stdout and stderr. The
//! child is `#[ignore]`d so a plain `cargo test` never runs it directly.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;

use kglite::api::io::{load_file, load_file_with, save_graph, LoadOptions};
use kglite::api::mutation::{add_nodes, ColumnData, ColumnType, DataFrame};
use kglite::api::{DirGraph, GraphRead};

/// Which scenario a child process is running.
const CASE_VAR: &str = "KGLITE_TEST_CEILING_CASE";
/// Where the child finds the fixture the parent wrote.
const FIXTURE_VAR: &str = "KGLITE_TEST_CEILING_FIXTURE";
const MAX_LOAD_VAR: &str = "KGLITE_MAX_LOAD_MB";

/// 2,000 `:Doc` rows with a string body — big enough that its estimate is well
/// over a zero ceiling and well under a 4 GB one, so both directions are
/// decided by the ceiling rather than by the fixture's size.
fn write_fixture(path: &Path) {
    let mut graph = DirGraph::new();
    let rows = 2_000;
    let body = "x".repeat(64);
    let mut df = DataFrame::new(Vec::new());
    df.add_column(
        "id".to_string(),
        ColumnType::String,
        ColumnData::String((0..rows).map(|i| Some(format!("n{i}"))).collect()),
    )
    .expect("id column");
    df.add_column(
        "body".to_string(),
        ColumnType::String,
        ColumnData::String((0..rows).map(|_| Some(body.clone())).collect()),
    )
    .expect("body column");
    add_nodes(
        &mut graph,
        df,
        "Doc".to_string(),
        "id".to_string(),
        None,
        None,
    )
    .expect("add nodes");
    let mut arc = Arc::new(graph);
    save_graph(&mut arc, path.to_str().unwrap()).expect("save fixture");
}

/// Run this binary's `child_scenario` with `case` and `max_load` in its
/// environment, and return what it printed.
fn run_child(case: &str, max_load: Option<&str>) -> (Output, PathBuf) {
    // The counter, not just the case name: three tests run the same case with
    // different environments and `cargo test` runs them in parallel, so a
    // shared directory had one test deleting another's fixture mid-run.
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "kglite_ceiling_env_{}_{case}_{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&dir).expect("scenario dir");
    let fixture = dir.join("graph.kgl");
    write_fixture(&fixture);

    let mut command = Command::new(std::env::current_exe().expect("current exe"));
    command
        .args(["--exact", "child_scenario", "--ignored", "--nocapture"])
        .env(CASE_VAR, case)
        .env(FIXTURE_VAR, &fixture);
    match max_load {
        Some(value) => command.env(MAX_LOAD_VAR, value),
        // Explicitly cleared: this test binary's own environment must not leak
        // a ceiling into a scenario that is asserting there is none.
        None => command.env_remove(MAX_LOAD_VAR),
    };
    let output = command.output().expect("spawn child");
    (output, dir)
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// The control: with no `KGLITE_MAX_LOAD_MB` at all, the same fixture loads —
/// so every refusal below is the variable's doing and not the fixture's.
#[test]
fn without_the_variable_there_is_no_ceiling() {
    let (output, dir) = run_child("plain_load", None);
    let stdout = stdout_of(&output);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(stdout.contains("RESULT: loaded 2000"), "{stdout}");
}

/// A ceiling the graph cannot fit under refuses the load, through the plain
/// `load_file` entry point that passes no options at all.
#[test]
fn the_variable_alone_refuses_a_plain_load() {
    let (output, dir) = run_child("plain_load", Some("0"));
    let stdout = stdout_of(&output);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(stdout.contains("RESULT: refused OutOfMemory"), "{stdout}");
    assert!(stdout.contains("estimated to peak at"), "{stdout}");
    assert!(stdout.contains(MAX_LOAD_VAR), "{stdout}");
}

/// A ceiling above the estimate is not a refusal — the variable is a limit, not
/// a switch.
#[test]
fn a_ceiling_above_the_estimate_loads() {
    let (output, dir) = run_child("plain_load", Some("4096"));
    let stdout = stdout_of(&output);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(stdout.contains("RESULT: loaded 2000"), "{stdout}");
}

/// An explicit option outranks the variable in the direction that matters most:
/// lifting a process-wide ceiling for one call that knows it can afford the
/// graph. Without this the variable would be unopposable from code.
#[test]
fn an_explicit_option_lifts_the_variables_ceiling() {
    let (output, dir) = run_child("option_lifts_ceiling", Some("0"));
    let stdout = stdout_of(&output);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(stdout.contains("RESULT: loaded 2000"), "{stdout}");
}

/// And in the other direction: an option imposes a ceiling where the
/// environment sets none.
#[test]
fn an_explicit_option_imposes_a_ceiling_without_the_variable() {
    let (output, dir) = run_child("option_imposes_ceiling", None);
    let stdout = stdout_of(&output);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(stdout.contains("RESULT: refused OutOfMemory"), "{stdout}");
}

/// **The loud half.** An unparseable value is treated as unset — refusing every
/// load over a typo would be worse, since the variable is process-wide — but it
/// must say so on stderr, naming the value. A silently ignored ceiling is the
/// exact failure the ceiling exists to prevent, and the first evidence would
/// otherwise be the OOM kill.
#[test]
fn an_unparseable_value_warns_loudly_and_lifts_the_ceiling() {
    let (output, dir) = run_child("plain_load", Some("1O24"));
    let stdout = stdout_of(&output);
    let stderr = stderr_of(&output);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(stdout.contains("RESULT: loaded 2000"), "{stdout}");
    assert!(
        stderr.contains(MAX_LOAD_VAR) && stderr.contains("1O24"),
        "the warning must name the variable and the value it could not read:\n{stderr}"
    );
    assert!(
        stderr.contains("NO memory ceiling"),
        "the warning must say what the operator lost:\n{stderr}"
    );
}

/// The scenario body, run in a child process. Ignored so it is only ever
/// reached through [`run_child`].
#[test]
#[ignore]
fn child_scenario() {
    let case = std::env::var(CASE_VAR).expect("child needs a case");
    let fixture = std::env::var(FIXTURE_VAR).expect("child needs a fixture");

    let loaded = match case.as_str() {
        // Whatever the environment says, through the entry point that takes no
        // options — the one every existing binding already calls.
        "plain_load" => load_file(&fixture),
        "option_lifts_ceiling" => {
            load_file_with(&fixture, &LoadOptions::new().with_max_load_bytes(None))
        }
        "option_imposes_ceiling" => {
            load_file_with(&fixture, &LoadOptions::new().with_max_load_bytes(Some(1)))
        }
        other => panic!("unknown case {other}"),
    };
    match loaded {
        Ok(graph) => println!("RESULT: loaded {}", graph.graph.node_count()),
        Err(error) => println!("RESULT: refused {:?} — {error}", error.kind()),
    }
}
