//! The interactive read-eval-print loop.
//!
//! Runs Cypher and a set of `sqlite3`-style dot-commands against an open
//! graph. Ctrl-C cancels a running query (the prompt itself stays alive —
//! rustyline keeps the terminal in raw mode, so SIGINT only fires while a
//! query is executing).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use kglite::api::io::{save_graph, to_csv_dir};
use kglite::api::session::{execute_mut, ExecuteOptions};
use kglite::api::{make_dir_graph_mut, DirGraph, Value};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::format::{render, Mode};

const PROMPT: &str = "kglite> ";

/// Set by the SIGINT handler; polled by the executor via `ExecuteOptions.cancel`.
/// `'static` because the only setter is a process-global signal handler.
static CANCEL: AtomicBool = AtomicBool::new(false);

const HELP: &str = "\
Commands:
  .help                  show this help
  .quit / .exit          leave the shell
  .labels                list node types          (CALL db.labels)
  .rels                  list relationship types  (CALL db.relationshipTypes)
  .schema                per-type property schema (CALL db.schema)
  .indexes               list indexes             (CALL db.indexes)
  .mode table|csv|json   set output format
  .dump <dir>            export a portable CSV + blueprint copy
  .read <file>           run the Cypher statements in a file
  .save [path]           save the graph to a .kgl file
Anything else is run as Cypher, e.g.
  MATCH (n) RETURN labels(n), count(*)
Ctrl-C cancels a running query; Ctrl-D (or .quit) exits.
Note: .import is not yet supported (no LOAD CSV) — use .read or .dump/from_blueprint.";

/// Mutable shell state threaded through the loop.
struct Shell {
    graph: Arc<DirGraph>,
    /// The file the graph is associated with (for `.save` with no argument).
    path: Option<String>,
    mode: Mode,
}

/// Run the REPL against `graph` until EOF / `.quit`.
pub fn run(graph: Arc<DirGraph>, source: Option<&str>) -> Result<()> {
    // Install the SIGINT → cancel handler once. Ignore an error (e.g. handler
    // already set in an odd embedding) — the shell still works, just without
    // mid-query cancellation.
    let _ = ctrlc::set_handler(|| CANCEL.store(true, Ordering::SeqCst));

    let mut rl = DefaultEditor::new()?;
    let mut shell = Shell {
        graph,
        path: source.map(str::to_string),
        mode: Mode::default(),
    };
    match &shell.path {
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
                    if !shell.dispatch_dot(cmd) {
                        break; // .quit / .exit
                    }
                } else {
                    shell.run_cypher(line);
                }
            }
            // Ctrl-C at the prompt: abandon the line, keep the session.
            Err(ReadlineError::Interrupted) => continue,
            // Ctrl-D: exit cleanly.
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }
    }
    Ok(())
}

impl Shell {
    /// Handle a `.`-prefixed command. Returns `false` to exit the loop.
    fn dispatch_dot(&mut self, cmd: &str) -> bool {
        let mut parts = cmd.split_whitespace();
        let name = parts.next().unwrap_or("");
        let arg = cmd[name.len()..].trim();
        match name {
            "help" | "h" | "?" => println!("{HELP}"),
            "quit" | "exit" | "q" => return false,
            "labels" => self.run_cypher("CALL db.labels() YIELD label RETURN label ORDER BY label"),
            "rels" | "relationshiptypes" => self.run_cypher(
                "CALL db.relationshipTypes() YIELD relationshipType \
                 RETURN relationshipType ORDER BY relationshipType",
            ),
            "schema" => self.run_cypher(
                "CALL db.schema() YIELD nodeType, properties \
                 RETURN nodeType, properties ORDER BY nodeType",
            ),
            "indexes" => self.run_cypher(
                "CALL db.indexes() YIELD name, type, properties \
                 RETURN name, type, properties ORDER BY name",
            ),
            "mode" => self.set_mode(arg),
            "dump" => self.dump(arg),
            "read" => self.read_file(arg),
            "save" => self.save(arg),
            "import" => {
                println!("`.import` is not supported (no LOAD CSV). Use .read <file.cypher>, or load a CSV+blueprint via kglite.from_blueprint().")
            }
            other => println!("Unknown command '.{other}'. Try .help."),
        }
        true
    }

    fn set_mode(&mut self, arg: &str) {
        match Mode::parse(arg) {
            Some(m) => {
                self.mode = m;
                println!("output mode: {}", m.name());
            }
            None => println!(
                "Usage: .mode table|csv|json (current: {})",
                self.mode.name()
            ),
        }
    }

    fn dump(&self, dir: &str) {
        if dir.is_empty() {
            println!("Usage: .dump <directory>");
            return;
        }
        match to_csv_dir(&self.graph, dir, None, &self.graph.parent_types) {
            Ok(summary) => {
                for line in &summary.log_lines {
                    println!("{line}");
                }
                println!("exported to {dir}/ — reload with kglite.from_blueprint('{dir}/blueprint.json')");
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }

    fn read_file(&mut self, file: &str) {
        if file.is_empty() {
            println!("Usage: .read <file>");
            return;
        }
        let contents = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: cannot read {file}: {e}");
                return;
            }
        };
        for stmt in split_statements(&contents) {
            println!("kglite> {stmt}");
            self.run_cypher(&stmt);
        }
    }

    fn save(&mut self, arg: &str) {
        let target = if arg.is_empty() {
            self.path.clone()
        } else {
            Some(arg.to_string())
        };
        let Some(path) = target else {
            println!("Usage: .save <path>  (no path is associated with this session yet)");
            return;
        };
        match save_graph(&mut self.graph, &path) {
            Ok(()) => {
                println!("saved to {path}");
                self.path = Some(path);
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// Execute one Cypher statement and print it in the active mode. Everything
    /// goes through `execute_mut`: a read query run that way executes as a read
    /// (no version bump), so one path serves reads and writes in this
    /// single-user shell.
    fn run_cypher(&mut self, query: &str) {
        let params: HashMap<String, Value> = HashMap::new();
        let mut opts = ExecuteOptions::new(&params);
        opts.cancel = Some(&CANCEL);
        CANCEL.store(false, Ordering::SeqCst);

        let g = make_dir_graph_mut(&mut self.graph);
        match execute_mut(g, query, &opts) {
            Ok(outcome) => {
                let r = &outcome.result;
                println!("{}", render(self.mode, &r.columns, &r.rows));
            }
            Err(e) => {
                if CANCEL.load(Ordering::SeqCst) {
                    println!("^C query cancelled");
                } else {
                    eprintln!("error: {e}");
                }
            }
        }
    }
}

/// Split a `.read` file into individual Cypher statements. Statements are
/// `;`-terminated when any `;` is present (the common `.cypher` convention);
/// otherwise each non-blank, non-`//`-comment line is one statement.
fn split_statements(contents: &str) -> Vec<String> {
    if contents.contains(';') {
        contents
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        contents
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty() && !s.starts_with("//"))
            .map(str::to_string)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::split_statements;

    #[test]
    fn split_on_semicolons() {
        let got = split_statements("CREATE (:A);  MATCH (n) RETURN n ;\n");
        assert_eq!(got, vec!["CREATE (:A)", "MATCH (n) RETURN n"]);
    }

    #[test]
    fn split_on_lines_when_no_semicolons() {
        let got = split_statements("MATCH (a) RETURN a\n// a comment\n\nMATCH (b) RETURN b\n");
        assert_eq!(got, vec!["MATCH (a) RETURN a", "MATCH (b) RETURN b"]);
    }
}
