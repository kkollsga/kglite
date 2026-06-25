//! The interactive read-eval-print loop.
//!
//! MVP scope (Phase 4): run Cypher, print an aligned table, plus `.help` /
//! `.quit`. Dot-commands (`.labels`, `.schema`, `.dump`, `.read`, `.mode`,
//! `.save`) and Ctrl-C query cancellation land in Phase 5.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use kglite::api::session::{execute_mut, ExecuteOptions};
use kglite::api::{make_dir_graph_mut, DirGraph, Value};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::format::render_table;

const PROMPT: &str = "kglite> ";

const HELP: &str = "\
Commands:
  .help            show this help
  .quit / .exit    leave the shell
Anything else is run as Cypher, e.g.
  MATCH (n) RETURN labels(n), count(*)
  CALL db.labels() YIELD label RETURN label";

/// Run the REPL against `graph` until EOF / `.quit`. `source` is the file the
/// graph came from (or `None` for a fresh in-memory graph) — shown in the
/// banner.
pub fn run(mut graph: Arc<DirGraph>, source: Option<&str>) -> Result<()> {
    let mut rl = DefaultEditor::new()?;
    match source {
        Some(p) => println!("kglite shell — {p}"),
        None => println!("kglite shell — in-memory graph (not saved)"),
    }
    println!("Type .help for commands, .quit to exit.");

    loop {
        match rl.readline(PROMPT) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(line);
                if let Some(cmd) = line.strip_prefix('.') {
                    if !dispatch_dot(cmd) {
                        break; // .quit / .exit
                    }
                } else {
                    run_cypher(&mut graph, line);
                }
            }
            // Ctrl-C: abandon the current line, keep the session (matches
            // sqlite3 / most REPLs). Phase 5 makes it cancel a running query.
            Err(ReadlineError::Interrupted) => continue,
            // Ctrl-D at the prompt: exit cleanly.
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }
    }
    Ok(())
}

/// Handle a `.`-prefixed command. Returns `false` to signal the loop to exit.
fn dispatch_dot(cmd: &str) -> bool {
    match cmd.split_whitespace().next().unwrap_or("") {
        "help" | "h" | "?" => println!("{HELP}"),
        "quit" | "exit" | "q" => return false,
        other => println!("Unknown command '.{other}'. Try .help."),
    }
    true
}

/// Execute one Cypher statement and print the table or the error. Everything
/// goes through `execute_mut`: a read query run that way executes as a read
/// (no version bump), so a single path serves reads and writes in this
/// single-user shell.
fn run_cypher(graph: &mut Arc<DirGraph>, query: &str) {
    let params: HashMap<String, Value> = HashMap::new();
    let opts = ExecuteOptions::new(&params);
    let g = make_dir_graph_mut(graph);
    match execute_mut(g, query, &opts) {
        Ok(outcome) => {
            let r = &outcome.result;
            println!("{}", render_table(&r.columns, &r.rows));
        }
        Err(e) => eprintln!("error: {e}"),
    }
}
